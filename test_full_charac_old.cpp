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
#include <limits>

// Third-party
// #include <fmt/base.h>

// tt-stl helpers
// #include <tt_stl/assert.hpp>
// #include <tt_stl/span.hpp>

// tt-metalium / platform
#include "tt_metal/tt_metal/perf_microbenchmark/common/util.hpp"
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/persistent_kernel_cache.hpp>

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

namespace {

void enable_persistent_kernel_cache_if_available() {
  tt::tt_metal::detail::EnablePersistentKernelCache();
}

void disable_persistent_kernel_cache_if_available() {
  tt::tt_metal::detail::DisablePersistentKernelCache();
}

} // namespace

////////////////////////////////////////////////////////////////////////////////////////
// Default values for benchmark parameters
////////////////////////////////////////////////////////////////////////////////////////

constexpr uint32_t DEFAULT_ITERATIONS = 10000;
constexpr uint32_t MAX_ARGS = 255;

enum class TestType : uint32_t {
  EmptyKernelLaunch = 0,
  ComputeMM = 1,
  SubDeviceMM = 2,
  HostPipelineComputeMM = 3,
  HostPipelineEmpty = 4,
  InvalidTest = 5
};

// Definition of test parameters structure
struct TestParams {
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t dtype; // 0: BFP8, 1: FP16
  uint32_t fidel; // 0: low, 1: high
  bool use_dram;
  bool use_cache;
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
  bool use_dram = false;
  bool use_cache = false;
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
    std::tie(use_dram, input_args) =
        test_args::has_command_option_and_remaining_args(input_args, "--dram");
    std::tie(use_cache, input_args) =
        test_args::has_command_option_and_remaining_args(input_args,
                                                         "--cache");

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
                                                                "--test", 5);
    if (test_uint > static_cast<uint32_t>(TestType::InvalidTest)) {
      throw std::runtime_error("Invalid --test value");
    }
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
                    .use_dram = use_dram,
                    .use_cache = use_cache,
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

std::tuple<uint32_t, uint32_t, uint32_t>
get_aligned_input_tile_num(uint32_t M, uint32_t N, uint32_t K) {
  auto align_to_tile = [](uint32_t value) -> uint32_t {
    return ((value + (constants::TILE_WIDTH - 1)) / constants::TILE_WIDTH) *
           constants::TILE_WIDTH;
  };

  uint32_t M_aligned = align_to_tile(M);
  uint32_t N_aligned = align_to_tile(N);
  uint32_t K_aligned = align_to_tile(K);

  return {M_aligned / constants::TILE_WIDTH, N_aligned / constants::TILE_WIDTH,
          K_aligned / constants::TILE_WIDTH};
}

std::vector<float> generate_fp32_random(uint32_t num_elems,
                                        float scale = 1.0f) {
  std::vector<float> vec(num_elems);
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  auto rand_float = std::bind(
      std::uniform_real_distribution<float>(-scale, scale), std::mt19937(seed));
  for (uint32_t i = 0; i < num_elems; ++i) {
    vec.at(i) = rand_float();
  }
  return vec;
}

float get_pcc(const std::vector<float> &x, const std::vector<float> &y) {
  if (x.size() != y.size() || x.empty()) {
    return 0.0f;
  }

  size_t n = x.size();
  double sum_x = 0.0, sum_y = 0.0;
  for (size_t i = 0; i < n; ++i) {
    sum_x += x[i];
    sum_y += y[i];
  }
  double mean_x = sum_x / n;
  double mean_y = sum_y / n;

  double numerator = 0.0;
  double sum_sq_diff_x = 0.0;
  double sum_sq_diff_y = 0.0;

  for (size_t i = 0; i < n; ++i) {
    double diff_x = x[i] - mean_x;
    double diff_y = y[i] - mean_y;
    numerator += diff_x * diff_y;
    sum_sq_diff_x += diff_x * diff_x;
    sum_sq_diff_y += diff_y * diff_y;
  }

  double denominator = std::sqrt(sum_sq_diff_x) * std::sqrt(sum_sq_diff_y);
  if (denominator == 0)
    return 0.0f;
  return static_cast<float>(numerator / denominator);
}

struct HostPipelineStats {
  double generate_us = 0.0;
  double transform_us = 0.0;
  double write_us = 0.0;
  double read_us = 0.0;
  double inverse_transform_us = 0.0;
  double end_to_end_us = 0.0;
  uint64_t bytes_written = 0;
  uint64_t bytes_read = 0;
};

void log_host_pipeline_stats(const std::string &tag, const HostPipelineStats &s,
                             uint32_t num_iters) {
  double iters = static_cast<double>(std::max(1u, num_iters));
  double avg_generate_us = s.generate_us / iters;
  double avg_transform_us = s.transform_us / iters;
  double avg_write_us = s.write_us / iters;
  double avg_read_us = s.read_us / iters;
  double avg_inverse_transform_us = s.inverse_transform_us / iters;
  double avg_end_to_end_us = s.end_to_end_us / iters;

  double write_seconds = s.write_us / 1e6;
  double read_seconds = s.read_us / 1e6;
  double write_gbps =
      (write_seconds > 0.0)
          ? (static_cast<double>(s.bytes_written) / write_seconds) / 1e9
          : 0.0;
  double read_gbps =
      (read_seconds > 0.0)
          ? (static_cast<double>(s.bytes_read) / read_seconds) / 1e9
          : 0.0;

  log_info(LogTest,
           "{} host-only pipeline avg(us): gen={:.2f}, xform={:.2f}, "
           "write={:.2f}, read={:.2f}, inv_xform={:.2f}, e2e={:.2f}",
           tag, avg_generate_us, avg_transform_us, avg_write_us, avg_read_us,
           avg_inverse_transform_us, avg_end_to_end_us);

  log_info(LogTest,
           "{} host-only transfer totals: write_bytes={}, read_bytes={}, "
           "write_bw={:.3f} GB/s, read_bw={:.3f} GB/s",
           tag, s.bytes_written, s.bytes_read, write_gbps, read_gbps);
}

bool test_compute_mm(tt::tt_metal::IDevice * /*device*/,
                     const TestParams & /*params*/) {
  log_error(LogTest,
            "ComputeMM legacy port is not implemented yet in "
            "test_full_charac_old.cpp (known gap). Use test_full_charac.cpp "
            "for current ComputeMM coverage.");
  return false;
}

bool test_sub_device_manager_mm(tt::tt_metal::IDevice * /*device*/,
                                const TestParams & /*params*/) {
  log_error(LogTest,
            "SubDeviceMM legacy port is not implemented yet in "
            "test_full_charac_old.cpp (known gap). Use test_full_charac.cpp "
            "for current SubDevice coverage.");
  return false;
}

bool test_host_pipeline_compute_mm(tt::tt_metal::IDevice *device,
                                   const TestParams &params) {
  bool pass = true;
  try {
    ZoneScopedN("ComputeMM Functional Blocks");
    if (params.num_iters == 0) {
      log_error(LogTest, "Host-only ComputeMM requires --num-iters > 0");
      return false;
    }
    if (params.dtype != 0) {
      log_error(LogTest,
                "Host-only ComputeMM currently supports only --dtype 0 "
                "(BFP8 path), requested dtype={}",
                params.dtype);
      return false;
    }

    uint32_t Mt = 0, Nt = 0, Kt = 0;
    uint32_t single_tile_size = 0;
    uint32_t in0_size_bytes = 0;
    uint32_t in1_size_bytes = 0;
    std::shared_ptr<tt_metal::Buffer> in0_buffer;
    std::shared_ptr<tt_metal::Buffer> in1_buffer;

    {
      ZoneScopedN("ComputeMM Input Data Processing");
      std::tie(Mt, Nt, Kt) =
        get_aligned_input_tile_num(params.M, params.N, params.K);

      tt::DataFormat pipeline_data_format = tt::DataFormat::Bfp8_b;
      single_tile_size = tt::tile_size(pipeline_data_format);
      uint32_t in0_num_tiles = Mt * Kt;
      uint32_t in1_num_tiles = Kt * Nt;
      in0_size_bytes = in0_num_tiles * single_tile_size;
      in1_size_bytes = in1_num_tiles * single_tile_size;

      in0_buffer = tt_metal::CreateBuffer(tt_metal::InterleavedBufferConfig{
        .device = device,
        .size = in0_size_bytes,
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM});

      in1_buffer = tt_metal::CreateBuffer(tt_metal::InterleavedBufferConfig{
        .device = device,
        .size = in1_size_bytes,
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM});
    }

    HostPipelineStats stats;
    uint32_t completed_iters = 0;

    for (uint32_t i = 0; i < params.num_iters; ++i) {
      ZoneScopedN("ComputeMM Host Dispatch Iteration");
      ZoneValue(i);
      auto t_iter_start = std::chrono::steady_clock::now();

      std::vector<float> in0_vec;
      std::vector<float> in1_vec;
      {
        ZoneScopedN("ComputeMM Host Prepare Inputs");
        auto t0 = std::chrono::steady_clock::now();
        in0_vec = generate_fp32_random(Mt * Kt * constants::TILE_HW, 5.0f);
        in1_vec = generate_fp32_random(Kt * Nt * constants::TILE_HW, 5.0f);
        auto t1 = std::chrono::steady_clock::now();
        stats.generate_us +=
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                .count();
      }

      std::vector<uint32_t> in0_packed;
      std::vector<uint32_t> in1_packed;
      {
        ZoneScopedN("ComputeMM Host Transform Inputs");
        auto t0 = std::chrono::steady_clock::now();
        auto in0_tilized = tilize_swizzled(in0_vec, Mt * 32, Kt * 32);
        in0_packed = pack_as_bfp8_tiles(tt::stl::make_const_span(in0_tilized),
                                        /*row_major_input=*/true,
                                        /*is_exp_a=*/false);

        auto in1_tilized = tilize_swizzled(in1_vec, Kt * 32, Nt * 32);
        in1_packed = pack_as_bfp8_tiles(tt::stl::make_const_span(in1_tilized),
                                        /*row_major_input=*/true,
                                        /*is_exp_a=*/false);
        auto t1 = std::chrono::steady_clock::now();
        stats.transform_us +=
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                .count();
      }

      {
        ZoneScopedN("ComputeMM Host Dispatch");
        {
          ZoneScopedN("ComputeMM Host Enqueue");
          auto t0 = std::chrono::steady_clock::now();
          tt_metal::detail::WriteToBuffer(in0_buffer, in0_packed);
          tt_metal::detail::WriteToBuffer(in1_buffer, in1_packed);
          auto t1 = std::chrono::steady_clock::now();
          stats.write_us +=
              std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                  .count();
          stats.bytes_written +=
              static_cast<uint64_t>(in0_size_bytes + in1_size_bytes);
        }

        std::vector<uint32_t> in0_readback;
        std::vector<uint32_t> in1_readback;
        {
          ZoneScopedN("ComputeMM Host FinishWait");
          auto t0 = std::chrono::steady_clock::now();
          tt_metal::detail::ReadFromBuffer(in0_buffer, in0_readback);
          tt_metal::detail::ReadFromBuffer(in1_buffer, in1_readback);
          auto t1 = std::chrono::steady_clock::now();
          stats.read_us +=
              std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                  .count();
          stats.bytes_read +=
              static_cast<uint64_t>(in0_size_bytes + in1_size_bytes);
        }

        std::vector<float> in0_roundtrip;
        std::vector<float> in1_roundtrip;
        {
          ZoneScopedN("ComputeMM Host Post Processing");
          {
            ZoneScopedN("ComputeMM Host Device Readback and Decode");
            auto t0 = std::chrono::steady_clock::now();
            auto in0_unpacked = unpack_bfp8_tiles_into_float_vec(
                in0_readback, /*row_major_output=*/true, /*is_exp_a=*/false);
            in0_roundtrip = untilize_swizzled(in0_unpacked, Mt * 32, Kt * 32);

            auto in1_unpacked = unpack_bfp8_tiles_into_float_vec(
                in1_readback, /*row_major_output=*/true, /*is_exp_a=*/false);
            in1_roundtrip = untilize_swizzled(in1_unpacked, Kt * 32, Nt * 32);
            auto t1 = std::chrono::steady_clock::now();
            stats.inverse_transform_us +=
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                    .count();
          }

          if (!params.bypass_check) {
            ZoneScopedN("ComputeMM Host Validation Metrics");
            float in0_pcc = get_pcc(in0_vec, in0_roundtrip);
            float in1_pcc = get_pcc(in1_vec, in1_roundtrip);
            if (in0_pcc < 0.99f || in1_pcc < 0.99f) {
              log_error(LogTest,
                        "Host-only ComputeMM roundtrip check failed at iter {}: "
                        "in0_pcc={:.4f}, in1_pcc={:.4f}",
                        i, in0_pcc, in1_pcc);
              pass = false;
              break;
            }
          }
        }
      }

      if (!pass) {
        break;
      }

      auto t_iter_end = std::chrono::steady_clock::now();
      stats.end_to_end_us +=
          std::chrono::duration_cast<std::chrono::microseconds>(t_iter_end -
                                                                 t_iter_start)
              .count();
      completed_iters++;
    }

    log_info(LogTest,
             "Host-only ComputeMM pipeline dims: Mt={}, Nt={}, Kt={}, "
             "completed_iters={}",
             Mt, Nt, Kt, completed_iters);
    log_host_pipeline_stats("ComputeMM", stats, completed_iters);
  } catch (const std::exception &e) {
    pass = false;
    log_error(LogTest, "{}", e.what());
  }
  return pass;
}

bool test_host_pipeline_empty_tensor(tt::tt_metal::IDevice *device,
                                     const TestParams &params) {
  bool pass = true;
  try {
    if (params.num_iters == 0) {
      log_error(LogTest, "Host-only Empty requires --num-iters > 0");
      return false;
    }
    if (params.dtype != 0) {
      log_error(LogTest,
                "Host-only Empty currently supports only --dtype 0 "
                "(BFP8 path), requested dtype={}",
                params.dtype);
      return false;
    }

    auto [Mt, Nt, Kt] =
        get_aligned_input_tile_num(params.M, params.N, params.K);
    (void)Kt;

    tt::DataFormat pipeline_data_format = tt::DataFormat::Bfp8_b;
    uint32_t single_tile_size = tt::tile_size(pipeline_data_format);
    uint32_t num_tiles = Mt * Nt;
    uint32_t tensor_size_bytes = num_tiles * single_tile_size;
    uint64_t expected_tensor_elems =
      static_cast<uint64_t>(Mt) * static_cast<uint64_t>(Nt) *
      static_cast<uint64_t>(constants::TILE_HW);

    auto tensor_buffer = tt_metal::CreateBuffer(tt_metal::InterleavedBufferConfig{
        .device = device,
        .size = tensor_size_bytes,
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM});

    HostPipelineStats stats;
    double transfer_window_us = 0.0;
    uint64_t transfer_window_bytes = 0;
    uint32_t completed_iters = 0;

    for (uint32_t i = 0; i < params.num_iters; ++i) {
      auto t_iter_start = std::chrono::steady_clock::now();

      std::vector<float> tensor_vec;
      {
        auto t0 = std::chrono::steady_clock::now();
        tensor_vec = generate_fp32_random(Mt * Nt * constants::TILE_HW, 5.0f);
        auto t1 = std::chrono::steady_clock::now();
        stats.generate_us +=
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                .count();
      }
      if (tensor_vec.size() != expected_tensor_elems) {
        log_error(LogTest,
                  "Host-only Empty tensor element count mismatch: got={}, "
                  "expected={} (Mt={}, Nt={}, TILE_HW={})",
                  tensor_vec.size(), expected_tensor_elems, Mt, Nt,
                  constants::TILE_HW);
        return false;
      }

      std::vector<uint32_t> packed;
      {
        auto t0 = std::chrono::steady_clock::now();
        auto tilized = tilize_swizzled(tensor_vec, Mt * 32, Nt * 32);
        packed = pack_as_bfp8_tiles(tt::stl::make_const_span(tilized),
                                    /*row_major_input=*/true,
                                    /*is_exp_a=*/false);
        auto t1 = std::chrono::steady_clock::now();
        stats.transform_us +=
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                .count();
      }
      uint64_t packed_bytes =
          static_cast<uint64_t>(packed.size()) * sizeof(uint32_t);
      if (packed_bytes != static_cast<uint64_t>(tensor_size_bytes)) {
        log_error(LogTest,
                  "Host-only Empty packed byte size mismatch: got={} B, "
                  "expected={} B (tiles={}, tile_size={})",
                  packed_bytes, tensor_size_bytes, num_tiles,
                  single_tile_size);
        return false;
      }

      auto t_transfer_start = std::chrono::steady_clock::now();
      {
        auto t0 = std::chrono::steady_clock::now();
        tt_metal::detail::WriteToBuffer(tensor_buffer, packed);
        auto t1 = std::chrono::steady_clock::now();
        stats.write_us +=
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                .count();
        stats.bytes_written += static_cast<uint64_t>(tensor_size_bytes);
      }

      std::vector<uint32_t> readback;
      {
        auto t0 = std::chrono::steady_clock::now();
        tt_metal::detail::ReadFromBuffer(tensor_buffer, readback);
        auto t1 = std::chrono::steady_clock::now();
        stats.read_us +=
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                .count();
        stats.bytes_read += static_cast<uint64_t>(tensor_size_bytes);
      }
      uint64_t readback_bytes =
          static_cast<uint64_t>(readback.size()) * sizeof(uint32_t);
      if (readback_bytes != static_cast<uint64_t>(tensor_size_bytes)) {
        log_error(LogTest,
                  "Host-only Empty readback byte size mismatch: got={} B, "
                  "expected={} B",
                  readback_bytes, tensor_size_bytes);
        return false;
      }
      auto t_transfer_end = std::chrono::steady_clock::now();
      transfer_window_us +=
          std::chrono::duration_cast<std::chrono::microseconds>(t_transfer_end -
                                                                 t_transfer_start)
              .count();
      transfer_window_bytes += static_cast<uint64_t>(tensor_size_bytes) * 2;

      std::vector<float> roundtrip;
      {
        auto t0 = std::chrono::steady_clock::now();
        auto unpacked = unpack_bfp8_tiles_into_float_vec(
            readback, /*row_major_output=*/true, /*is_exp_a=*/false);
        roundtrip = untilize_swizzled(unpacked, Mt * 32, Nt * 32);
        auto t1 = std::chrono::steady_clock::now();
        stats.inverse_transform_us +=
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                .count();
      }

      if (!params.bypass_check) {
        ZoneScopedN("HostPipeline Empty Validation Metrics");
        float pcc = get_pcc(tensor_vec, roundtrip);
        if (pcc < 0.99f) {
          log_error(LogTest,
                    "Host-only Empty roundtrip check failed at iter {}: "
                    "pcc={:.4f}",
                    i, pcc);
          pass = false;
          break;
        }
      }

      auto t_iter_end = std::chrono::steady_clock::now();
      stats.end_to_end_us +=
          std::chrono::duration_cast<std::chrono::microseconds>(t_iter_end -
                                                                 t_iter_start)
              .count();
      completed_iters++;
    }

    log_info(LogTest,
             "Host-only Empty pipeline dims: Mt={}, Nt={}, completed_iters={}",
             Mt, Nt, completed_iters);
    double transfer_seconds = transfer_window_us / 1e6;
    double effective_bw_mb_s =
      (transfer_seconds > 0.0)
        ? (static_cast<double>(transfer_window_bytes) / transfer_seconds) /
            1e6
        : 0.0;
    log_info(LogTest,
         "Host-only Empty effective transfer BW: total_bytes={}, "
         "transfer_time={:.2f}us, bw={:.3f} MB/s",
         transfer_window_bytes, transfer_window_us, effective_bw_mb_s);
    log_host_pipeline_stats("Empty", stats, completed_iters);
  } catch (const std::exception &e) {
    pass = false;
    log_error(LogTest, "{}", e.what());
  }
  return pass;
}

///////////////////////////////////////////////////////////////////////////////////////////
/// Empty Kernel Launch Test
///////////////////////////////////////////////////////////////////////////////////////////
bool test_empty_kernel_launch(tt::tt_metal::IDevice *device,
                              const TestParams &params) {
  bool pass = true;
  try {
    ZoneScopedN("EmptyKernel Functional Blocks");
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program = tt_metal::Program();
    uint32_t single_tile_size = 2 * 1024;
    // std::vector<unsigned long> elapsed_us;

    {
      ZoneScopedN("EmptyKernel Host Setup");
      for (int core_group_idx = 0; core_group_idx < params.core_groups;
           ++core_group_idx) {
        CoreCoord start_core = {0, (params.core_y / params.core_groups) *
                                       core_group_idx};
        CoreCoord end_core = {
            (std::size_t)params.core_x - 1,
            (core_group_idx == params.core_groups - 1)
                ? (std::size_t)params.core_y - 1
                : ((params.core_y / params.core_groups) *
                   (core_group_idx + 1)) -
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
    // auto t_compile_begin = std::chrono::steady_clock::now();
    // tt_metal::detail::CompileProgram(device, program);
    // auto t_compile_end = std::chrono::steady_clock::now();
    // auto compile_time = std::chrono::duration_cast<std::chrono::microseconds>(
    //                         t_compile_end - t_compile_begin)
    //                         .count();
    // log_info(LogTest, "Time elapsed for compilation: {}us", compile_time);

    // Now we should have a cache hit
    log_info(LogTest, "Num tests {}", params.num_iters);
    {
      ZoneScopedN("EmptyKernel Host Dispatch");
      for (uint32_t i = 0; i < params.num_iters; ++i) {
        ZoneScopedN("EmptyKernel Host Dispatch Iteration");
        ZoneValue(i);
        {
          ZoneScopedN("EmptyKernel Host Enqueue");
          EnqueueProgram(device->command_queue(), program, false);
        }
        {
          ZoneScopedN("EmptyKernel Host FinishWait");
          Finish(device->command_queue());
        }
        // auto t_begin = std::chrono::steady_clock::now();
        // auto t_end = std::chrono::steady_clock::now();
        // elapsed_us.push_back(
        //     std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_begin)
        //         .count());
        //
        // log_info(LogTest, "Time elapsed for executing empty kernels: {}us",
        //          elapsed_us[i]);
      }
    }

    // Calculate stats
    // std::sort(elapsed_us.begin(), elapsed_us.end());

    // Filter outliers if we have enough data (e.g. > 2 samples)
    // std::vector<unsigned long> filtered_elapsed_us;
    // if (elapsed_us.size() > 2) {
    //   // Exclude min and max
    //   filtered_elapsed_us.assign(elapsed_us.begin() + 1, elapsed_us.end() - 1);
    // } else {
    //   filtered_elapsed_us = elapsed_us;
    // }

    // auto min_val = *std::min_element(filtered_elapsed_us.begin(),
    //                                  filtered_elapsed_us.end());
    // auto max_val = *std::max_element(filtered_elapsed_us.begin(),
    //                                  filtered_elapsed_us.end());
    // auto sum_val = std::accumulate(filtered_elapsed_us.begin(),
    //                                filtered_elapsed_us.end(), 0.0);
    // auto avg_val = sum_val / filtered_elapsed_us.size();

    // double sum_sq_diff = 0.0;
    // for (const auto &val : filtered_elapsed_us) {
    //   double diff = val - avg_val;
    //   sum_sq_diff += diff * diff;
    // }
    // auto std_dev = std::sqrt(sum_sq_diff / filtered_elapsed_us.size());
    // log_info(LogTest,
    //          "Execution Stats (us) [Trimmed]: Min={}, Max={}, Avg={:.2f}, "
    //          "StdDev={:.2f}",
    //          min_val, max_val, avg_val, std_dev);

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

  // Parse input arguments
  DeviceParams device_params;
  std::vector<std::string> input_args(argv, argv + argc);
  TestParams params = parse_input_arguments(input_args, device_params);

  if (params.use_cache) {
    if (params.clean_mode == 1) {
      log_warning(LogTest,
                  "--cache with --clean-mode 1 will invalidate disk cache "
                  "benefits for this run.");
    }
    enable_persistent_kernel_cache_if_available();
  }

  // Create device
  int device_id = 0;
  device_params.device = tt_metal::CreateDevice(device_id);
  device_params.grid_coord =
      device_params.device->compute_with_storage_grid_size();

  // Definition of max cores in each dimension for the architecture being tested
  uint32_t max_x = device_params.grid_coord.x;
  uint32_t max_y = device_params.grid_coord.y;

  if (params.cpu_id != 0xFFFFFFFF) {
    pin_to_cpu(params.cpu_id);
  }

  //// Print test summary
  log_info(LogTest, "Full Characterization Benchmarking Test (LEGACY)");
  log_info(LogTest, "==============================================");
  log_info(LogTest, "Selectec Test: {}", static_cast<uint32_t>(params.test));
  log_info(LogTest,
           "Starting with parameters: M={}, N={}, K={}, dtype={}, fidel={}, "
           "core_x={}, core_y={}, core_groups={}, num_iters={}, clean_mode={}, "
           "cache={}",
           params.M, params.N, params.K, params.dtype, params.fidel,
           params.core_x, params.core_y, params.core_groups, params.num_iters,
           params.clean_mode, params.use_cache);

  if (params.core_x > max_x || params.core_y > max_y) {
    log_error(
        tt::LogTest,
        "Requested core size ({},{}) exceeds max for architecture ({},{})",
        params.core_x, params.core_y, max_x, max_y);
    if (params.use_cache) {
      disable_persistent_kernel_cache_if_available();
    }
    tt_metal::CloseDevice(device_params.device);
    return -1;
  }

  bool pass = false;
  if (params.use_cache) {
    device_params.device->enable_program_cache();
    log_info(LogTest,
             "Persistent kernel cache + program cache enabled (--cache)");
  }

  switch (params.test) {
  case TestType::EmptyKernelLaunch:
    pass = test_empty_kernel_launch(device_params.device, params);
    break;
  case TestType::ComputeMM:
    pass = test_compute_mm(device_params.device, params);
    break;
  case TestType::SubDeviceMM:
    pass = test_sub_device_manager_mm(device_params.device, params);
    break;
  case TestType::HostPipelineComputeMM:
    pass = test_host_pipeline_compute_mm(device_params.device, params);
    break;
  case TestType::HostPipelineEmpty:
    pass = test_host_pipeline_empty_tensor(device_params.device, params);
    break;
  default:
    log_error(tt::LogTest, "Invalid test type selected: {}",
              static_cast<uint32_t>(params.test));
    if (params.use_cache) {
      device_params.device->disable_and_clear_program_cache();
      disable_persistent_kernel_cache_if_available();
      log_info(LogTest,
               "Program cache disabled/cleared and persistent kernel cache "
               "disabled");
    }
    return -1;
  }

  if (params.use_cache) {
    device_params.device->disable_and_clear_program_cache();
    disable_persistent_kernel_cache_if_available();
    log_info(LogTest,
             "Program cache disabled/cleared and persistent kernel cache "
             "disabled");
  }

  // We finalize the device
  tt_metal::CloseDevice(device_params.device);

  return 0;
}