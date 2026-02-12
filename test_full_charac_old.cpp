/*
 * FULL CHARACTERIZATION BENCHMARKING TEST (LEGACY PORT)
 *
 * The purpose of this test is to perform a full characterization benchmarking
 * of dispatching, data movement, compile times across a wide range of scenarios
 * and configurations. The main arguments to vary are: Grid Size (number of
 * cores) with x,y dimensions: --y_size, --x_size (minumum 0,0 to maximum of the
 * architecture)
 *
 */
/*  THIS BENCHMARK:
 *   +----------------------------+------+------+------+------+------------+
 *   | Test                       | Comp | Disp | Data | Comp | Tunable    |
 *   | Name                       | Time | Time | Trns | Real | Args       |
 *   +----------------------------+------+------+------+------+------------+
 *   | 7_kernel_launch            |  N   |  Y   |  N   |  N   | N (Hard)   |
 *   | 6_dram_offchip             |  N   |  Y   |  Y   |  N   | N          |
 *   | dispatch/test_pgm_dispatch |  N   |  Y   |  N   |  N   | Y          |
 *   | 1_compute_mm               |  N   |  Y   |  Y   |  Y   | N          |
 *   | This Benchmark             |  Y   |  Y   |  Y   |  Y   | Y          |
 *   +----------------------------+------+------+------+------+------------+
 */

// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// C / STL
#include <algorithm>
#include <array>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <ratio>
#include <sched.h>
#include <set>
#include <stdbool.h>
#include <stdint.h>
#include <string>
#include <tuple>
#include <type_traits>
#include <unistd.h>
#include <utility>
#include <variant>
#include <vector>

// Third-party
// #include <fmt/base.h>

// tt-stl helpers
// #include <tt_stl/assert.hpp>
// #include <tt_stl/span.hpp>

// tt-metalium / platform
#include "tt_metal/tt_metal/perf_microbenchmark/common/util.hpp"
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>

// Utility / test helpers
// #include "test_common.hpp"
#include "hostdevcommon/common_values.hpp"
#include "hostdevcommon/kernel_structs.h"
#include "tt_metal/tt_metal/perf_microbenchmark/common/util.hpp"
// #include "tt_metal/test_utils/deprecated/tensor.hpp"

// Impl / dispatch / internal
// #include "impl/dispatch/command_queue.hpp"
// #include "impl/context/metal_context.hpp"
// #include "impl/buffers/semaphore.hpp"
// #include "tt_metal/impl/dispatch/device_command.hpp"
// #include <impl/dispatch/dispatch_mem_map.hpp>

// Platform/device specific
// #include <umd/device/types/arch.hpp>
// #include <umd/device/types/xy_pair.hpp>

// Logging
// #include <tt-logger/tt-logger.hpp>

using namespace tt;
using namespace tt::tt_metal;

////////////////////////////////////////////////////////////////////////////////////////
// Default values for benchmark parameters
////////////////////////////////////////////////////////////////////////////////////////

constexpr uint32_t DEFAULT_ITERATIONS = 10000;
constexpr uint32_t MAX_ARGS = 255;

enum class TestType : uint32_t {
  EmptyKernelLaunch = 0,
  ReadKernelLaunch = 1,
  WriteKernelLaunch = 2,
  RWkernelLaunch = 3,
  InvalidTest = 4
};

// Definition of test parameters structure
struct TestParams {
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t dtype; // 0: BFP8, 1: FP16
  uint32_t fidel; // 0: low, 1: high
  uint32_t core_x;
  uint32_t core_y;
  uint32_t core_groups;
  uint32_t num_iters;
  bool bypass_check;
  uint32_t num_rt_args;
  uint32_t cpu_id;
  uint32_t clean_mode;
  TestType test;
};

struct DeviceParams {
  tt::tt_metal::IDevice *device;
  CoreCoord grid_coord;
};

///////////////////////////////////////////////////////////////////////////////////////////
/// Input Argument Parsing
///////////////////////////////////////////////////////////////////////////////////////////
TestParams parse_input_arguments(std::vector<std::string> input_args,
                                 DeviceParams &device_params) {
  uint32_t M, N, K;
  uint32_t dtype; // 0: BFP8, 1: FP16
  uint32_t fidel; // 0: low, 1: high
  uint32_t core_x, core_y, core_groups;
  uint32_t num_iters;
  bool bypass_check = false;
  uint32_t num_rt_args;
  uint32_t cpu_id;
  uint32_t clean_mode;
  TestType test;
  uint32_t test_uint;
  try {
    // Matrix related args
    std::tie(M, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(input_args,
                                                                "--m", 11264);
    std::tie(N, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(input_args,
                                                                "--n", 3072);
    std::tie(K, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(input_args,
                                                                "--k", 768);
    std::tie(dtype, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(input_args,
                                                                "--dtype", 0);
    std::tie(fidel, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(input_args,
                                                                "--fidel", 0);

    // Core grid size args
    std::tie(core_x, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(input_args,
                                                                "--x_size", 0);
    std::tie(core_y, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(input_args,
                                                                "--y_size", 0);
    std::tie(core_groups, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(
            input_args, "--core_groups", 1);

    // Benchmarking args
    std::tie(num_iters, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(
            input_args, "--num-iters", 15);
    std::tie(bypass_check, input_args) =
        test_args::has_command_option_and_remaining_args(input_args,
                                                         "--bypass-check");
    std::tie(num_rt_args, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(
            input_args, "--num-rt-args", 255);
    std::tie(cpu_id, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(
            input_args, "--cpu", 0xFFFFFFFF);

    std::tie(clean_mode, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(
            input_args, "--clean-mode", 0);

    std::tie(test_uint, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(input_args,
                                                                "--test", 4);
    test = static_cast<TestType>(test_uint);
    test_args::validate_remaining_args(input_args);

  } catch (const std::exception &e) {
    log_error(LogTest, "Command line arguments found exception", e.what());
    throw;
  }

  if (core_x == 0 && core_y == 0) {
    log_info(LogTest, "Running on single core (defaulting 0,0 to 1,1)");
    core_x = 1;
    core_y = 1;
  }

  if (core_groups == 0) {
    log_error(LogTest, "The number of core groups must be greater than 0");
    throw std::runtime_error("core_groups must be > 0");
  }

  if (core_y < core_groups) {
    log_error(
        LogTest,
        "The number of cores in a row ({}) must be bigger than or equal than "
        "the number of core groups ({})",
        core_y, core_groups);
    throw std::runtime_error("core_y < core_groups");
  }

  TestParams params{.M = M,
                    .N = N,
                    .K = K,
                    .dtype = dtype,
                    .fidel = fidel,
                    .core_x = core_x,
                    .core_y = core_y,
                    .core_groups = core_groups,
                    .num_iters = num_iters,
                    .bypass_check = bypass_check,
                    .num_rt_args = num_rt_args,
                    .cpu_id = cpu_id,
                    .clean_mode = clean_mode,
                    .test = test};
  return params;
}

///////////////////////////////////////////////////////////////////////////////////////////
/// Empty Kernel Launch Test
///////////////////////////////////////////////////////////////////////////////////////////
bool test_empty_kernel_launch(tt::tt_metal::IDevice *device,
                              const TestParams &params) {
  bool pass = true;
  try {
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program = tt_metal::Program();
    uint32_t single_tile_size = 2 * 1024;
    std::vector<unsigned long> elapsed_us;

    for (int core_group_idx = 0; core_group_idx < params.core_groups;
         ++core_group_idx) {
      CoreCoord start_core = {0, (params.core_y / params.core_groups) *
                                     core_group_idx};
      CoreCoord end_core = {
          (std::size_t)params.core_x - 1,
          (core_group_idx == params.core_groups - 1)
              ? (std::size_t)params.core_y - 1
              : ((params.core_y / params.core_groups) * (core_group_idx + 1)) -
                    1};
      CoreRange group_of_cores(start_core, end_core);

      log_info(
          LogTest, "Setting kernels for core group {}, cores ({},{}) ~ ({},{})",
          core_group_idx, start_core.x, start_core.y, end_core.x, end_core.y);

      for (int i = start_core.y; i <= end_core.y; i++) {
        for (int j = start_core.x; j <= end_core.x; j++) {
          CoreCoord core = {(std::size_t)j, (std::size_t)i};
          uint32_t cb_index = 0;
          uint32_t cb_tiles = 8;
          tt_metal::CircularBufferConfig cb_config =
              tt_metal::CircularBufferConfig(
                  cb_tiles * single_tile_size,
                  {{cb_index, tt::DataFormat::Float16_b}})
                  .set_page_size(cb_index, single_tile_size);
          tt_metal::CreateCircularBuffer(program, core, cb_config);
        }
      }

      std::vector<uint32_t> reader_compile_args = {uint32_t(core_group_idx)};
      auto reader_kernel = tt_metal::CreateKernel(
          program,
          "tests/tt_metal/tt_metal/perf_microbenchmark/13_full_charac/"
          "kernels/"
          "empty_reader.cpp",
          group_of_cores,
          tt_metal::DataMovementConfig{
              .processor = tt_metal::DataMovementProcessor::RISCV_1,
              .noc = tt_metal::NOC::RISCV_1_default,
              .compile_args = reader_compile_args});

      std::vector<uint32_t> writer_compile_args = {uint32_t(core_group_idx)};
      auto writer_kernel = tt_metal::CreateKernel(
          program,
          "tests/tt_metal/tt_metal/perf_microbenchmark/13_full_charac/"
          "kernels/"
          "empty_writer.cpp",
          group_of_cores,
          tt_metal::DataMovementConfig{
              .processor = tt_metal::DataMovementProcessor::RISCV_0,
              .noc = tt_metal::NOC::RISCV_0_default,
              .compile_args = writer_compile_args});

      std::vector<uint32_t> compute_compile_args = {uint32_t(core_group_idx)};
      tt_metal::CreateKernel(
          program,
          "tests/tt_metal/tt_metal/perf_microbenchmark/13_full_charac/"
          "kernels/"
          "empty_compute.cpp",
          group_of_cores,
          tt_metal::ComputeConfig{.compile_args = compute_compile_args});

      for (int i = start_core.y; i <= end_core.y; i++) {
        for (int j = start_core.x; j <= end_core.x; j++) {
          CoreCoord core = {(std::size_t)j, (std::size_t)i};
          int core_index = (i * params.core_x) + j;

          std::vector<uint32_t> reader_runtime_args(params.num_rt_args);
          std::vector<uint32_t> writer_runtime_args(params.num_rt_args);
          for (uint32_t k = 0; k < params.num_rt_args; ++k) {
            reader_runtime_args[k] = core_index + k;
            writer_runtime_args[k] = core_index + k;
          }

          SetRuntimeArgs(program, writer_kernel, core, writer_runtime_args);
          SetRuntimeArgs(program, reader_kernel, core, reader_runtime_args);
        }
      }
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////

    // Explicitly compile the program to measure compile overhead
    auto t_compile_begin = std::chrono::steady_clock::now();
    tt_metal::detail::CompileProgram(device, program);
    auto t_compile_end = std::chrono::steady_clock::now();
    auto compile_time = std::chrono::duration_cast<std::chrono::microseconds>(
                            t_compile_end - t_compile_begin)
                            .count();
    log_info(LogTest, "Time elapsed for compilation: {}us", compile_time);

    // Now we should have a cache hit
    log_info(LogTest, "Num tests {}", params.num_iters);
    for (uint32_t i = 0; i < params.num_iters; ++i) {
      auto t_begin = std::chrono::steady_clock::now();
      EnqueueProgram(device->command_queue(), program, false);
      Finish(device->command_queue());
      auto t_end = std::chrono::steady_clock::now();
      elapsed_us.push_back(
          std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_begin)
              .count());

      log_info(LogTest, "Time elapsed for executing empty kernels: {}us",
               elapsed_us[i]);
    }

    // Calculate stats
    std::sort(elapsed_us.begin(), elapsed_us.end());

    // Filter outliers if we have enough data (e.g. > 2 samples)
    std::vector<unsigned long> filtered_elapsed_us;
    if (elapsed_us.size() > 2) {
      // Exclude min and max
      filtered_elapsed_us.assign(elapsed_us.begin() + 1, elapsed_us.end() - 1);
    } else {
      filtered_elapsed_us = elapsed_us;
    }

    auto min_val = *std::min_element(filtered_elapsed_us.begin(),
                                     filtered_elapsed_us.end());
    auto max_val = *std::max_element(filtered_elapsed_us.begin(),
                                     filtered_elapsed_us.end());
    auto sum_val = std::accumulate(filtered_elapsed_us.begin(),
                                   filtered_elapsed_us.end(), 0.0);
    auto avg_val = sum_val / filtered_elapsed_us.size();

    double sum_sq_diff = 0.0;
    for (const auto &val : filtered_elapsed_us) {
      double diff = val - avg_val;
      sum_sq_diff += diff * diff;
    }
    auto std_dev = std::sqrt(sum_sq_diff / filtered_elapsed_us.size());
    log_info(LogTest,
             "Execution Stats (us) [Trimmed]: Min={}, Max={}, Avg={:.2f}, "
             "StdDev={:.2f}",
             min_val, max_val, avg_val, std_dev);

  } catch (const std::exception &e) {
    pass = false;
    log_error(LogTest, "{}", e.what());
    log_error(LogTest, "System error message: {}", std::strerror(errno));
  }
  return pass;
}

void pin_to_cpu(int cpu) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  if (sched_setaffinity(0, sizeof(set), &set) != 0) {
    log_warning(tt::LogTest, "Failed to pin to CPU {}: {}", cpu,
                std::strerror(errno));
  } else {
    log_info(tt::LogTest, "Pinned to CPU {}", cpu);
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
// Main Benchmarking Test
//////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
  // Tracy is enabled via environment variables (same strategy as new test)

  if (std::getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
    log_error(tt::LogTest, "Test not supported w/ slow dispatch, exiting");
  }

  // Create device
  int device_id = 0;
  DeviceParams device_params;

  device_params.device = tt_metal::CreateDevice(device_id);
  device_params.grid_coord =
      device_params.device->compute_with_storage_grid_size();

  // Definition of max cores in each dimension for the architecture being tested
  uint32_t max_x = device_params.grid_coord.x;
  uint32_t max_y = device_params.grid_coord.y;

  // Parse input arguments
  std::vector<std::string> input_args(argv, argv + argc);
  TestParams params = parse_input_arguments(input_args, device_params);

  if (params.cpu_id != 0xFFFFFFFF) {
    pin_to_cpu(params.cpu_id);
  }

  //// Print test summary
  log_info(LogTest, "Full Characterization Benchmarking Test (LEGACY)");
  log_info(LogTest, "==============================================");
  log_info(LogTest, "Selectec Test: {}", static_cast<uint32_t>(params.test));
  log_info(LogTest,
           "Starting with parameters: M={}, N={}, K={}, dtype={}, fidel={}, "
           "core_x={}, core_y={}, core_groups={}, num_iters={}, clean_mode={}",
           params.M, params.N, params.K, params.dtype, params.fidel,
           params.core_x, params.core_y, params.core_groups, params.num_iters,
           params.clean_mode);

  if (params.core_x > max_x || params.core_y > max_y) {
    log_error(
        tt::LogTest,
        "Requested core size ({},{}) exceeds max for architecture ({},{})",
        params.core_x, params.core_y, max_x, max_y);
    return -1;
  }

  bool pass = false;
  switch (params.test) {
  case TestType::EmptyKernelLaunch:
    pass = test_empty_kernel_launch(device_params.device, params);
    break;
  default:
    log_error(tt::LogTest, "Invalid test type selected: {}",
              static_cast<uint32_t>(params.test));
    return -1;
  }

  // We finalize the device
  tt_metal::CloseDevice(device_params.device);

  return 0;
}