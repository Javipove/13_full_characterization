/*
 * FULL CHARACTERIZATION BENCHMARKING TEST
 *
 * The purpose of this test is to perform a full characterization benchmarking
 * of dispatching, data movement, compile times across a wide range of scenarios
 * and configurations. The main arguments to vary are: Grid Size (number of
 * cores) with x,y dimensions: --y_size, --x_size (minumum 0,0 to maximum of the
 * architecture)
 *
 */
/*  THIS BENCHMARCH:
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
#include <filesystem>
#include <functional>
#include <limits>
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
// #include <benchmark/benchmark.h>
#include <fmt/base.h>

// tt-stl helpers
#include <tt_stl/assert.hpp>
#include <tt_stl/span.hpp>

// tt-metalium / platform
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/base_types.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
// command_queue.hpp removed upstream; APIs now in mesh_command_queue.hpp / distributed.hpp
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_command_queue.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/persistent_kernel_cache.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include <tt-metalium/work_split.hpp>

// Utility / test helpers
#include "hostdevcommon/common_values.hpp"
#include "hostdevcommon/kernel_structs.h"
#include "test_common.hpp"
#include "tt_metal/test_utils/deprecated/tensor.hpp"
#include "tt_metal/tt_metal/perf_microbenchmark/common/util.hpp"

// Impl / dispatch / internal
// #include "impl/dispatch/command_queue.hpp"
// #include "impl/context/metal_context.hpp"
// #include "impl/buffers/semaphore.hpp"
// #include "tt_metal/impl/dispatch/device_command.hpp"
// #include <impl/dispatch/dispatch_mem_map.hpp>

// Platform/device specific
#include <umd/device/types/arch.hpp>
#include <umd/device/types/xy_pair.hpp>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/mesh_buffer.hpp>

// Tracy
#include <tracy/Tracy.hpp>
#include <tt-metalium/tilize_utils.hpp>

// ttnn
// #include <ttnn/distributed/distributed.hpp>
#include <ttnn/operations/core/core.hpp> // Covers to_device, to_layout
#include <ttnn/operations/copy/typecast/typecast.hpp>
#include <ttnn/operations/creation.hpp>  // Covers empty()
#include <ttnn/operations/data_movement/tilize/tilize.hpp>
#include <ttnn/operations/data_movement/untilize/untilize.hpp>
#include <ttnn/tensor/tensor.hpp>        // Core Tensor class
#include <ttnn/tensor/tensor_utils.hpp> // Required for host-side data conversion helpers

using namespace tt;
using namespace tt::tt_metal;

namespace {

void enable_persistent_kernel_cache_if_available() {
  tt::tt_metal::detail::EnablePersistentKernelCache();
}

void disable_persistent_kernel_cache_if_available() {
  tt::tt_metal::detail::DisablePersistentKernelCache();
}

void log_validation_sample_pairs(const std::string &tag,
                                 const std::vector<float> &reference,
                                 const std::vector<float> &observed,
                                 size_t sample_count = 12) {
  size_t common_size = std::min(reference.size(), observed.size());
  size_t num_samples = std::min(sample_count, common_size);
  log_info(LogTest,
           "{} visual sample check (showing {} of {} aligned elements)", tag,
           num_samples, common_size);

  for (size_t i = 0; i < num_samples; ++i) {
    float abs_err = std::fabs(reference[i] - observed[i]);
    log_info(LogTest, "{} [{}] ref={:.6f}, obs={:.6f}, abs_err={:.6f}", tag, i,
             reference[i], observed[i], abs_err);
  }
}

std::string resolve_kernel_source_path(const std::string &kernel_filename) {
  namespace fs = std::filesystem;
  std::vector<fs::path> kernel_dirs;

  // Local project layout only: kernels/ lives next to this benchmark source
  // and run script. We intentionally do not look under TT_METAL_HOME.
  kernel_dirs.emplace_back(fs::current_path() / "kernels");

  std::error_code ec;
  fs::path exe_path = fs::read_symlink("/proc/self/exe", ec);
  if (!ec) {
    const fs::path exe_dir = exe_path.parent_path();
    kernel_dirs.emplace_back(exe_dir / "../kernels");
    kernel_dirs.emplace_back(exe_dir / "kernels");
  }

  for (const auto &dir : kernel_dirs) {
    ec.clear();
    if (!fs::exists(dir, ec) || !fs::is_directory(dir, ec)) {
      continue;
    }

    fs::path candidate = dir / kernel_filename;
    if (fs::exists(candidate, ec) && fs::is_regular_file(candidate, ec)) {
      fs::path canonical = fs::weakly_canonical(candidate, ec);
      return ec ? candidate.string() : canonical.string();
    }
  }

  throw std::runtime_error(
      "Unable to find kernel source file '" + kernel_filename +
      "'. Checked local cwd/kernels and executable-adjacent kernels "
      "directories.");
}

} // namespace

////////////////////////////////////////////////////////////////////////////////////////
// Default values for benchmark parameters
////////////////////////////////////////////////////////////////////////////////////////

// constexpr uint32_t DEFAULT_ITERATIONS = 10000;
// constexpr uint32_t DEFAULT_WARMUP_ITERATIONS = 100;
// constexpr uint32_t MIN_KERNEL_SIZE_BYTES = 32;  // overhead
// constexpr uint32_t DEFAULT_KERNEL_SIZE_K = 1;
// constexpr uint32_t MAX_CBS = 32;
// constexpr uint32_t MAX_ARGS = 255;

enum class TestType : uint32_t {
  EmptyKernelLaunch = 0,
  ComputeMM = 1,
  SubDeviceMM = 2,
  HostPipelineComputeMM = 3,
  HostPipelineEmpty = 4,
  ComputeMMAsyncBatch = 5,
  ComputeMMTraceReplay = 6,
  InvalidTest = 7
};

enum class OutputDTypeMode : uint32_t {
  Native = 0,
  Bf16 = 1,
  Fp32 = 2,
};

// Definition of test parameters structure
struct TestParams {
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t dtype; // Legacy selector: 0 BFP8, 1 BF16, 2 FP32 (mapped)
  uint32_t input_dtype; // Effective input selector for compute path
  OutputDTypeMode output_dtype_mode;
  uint32_t fidel; // 0: low, 1: high
  bool use_dram;  // Enable DRAM input buffers
  bool use_cache; // Enable program cache
  uint32_t core_x;
  uint32_t core_y;
  uint32_t core_groups;
  uint32_t num_iters;
  bool bypass_check;
  uint32_t num_rt_args;
  uint32_t trace_capture_ops;
  uint32_t cpu_id;
  uint32_t cpu_range;
  uint32_t clean_mode;
  TestType test;
  bool pack_device;   // true: device, false: cpu
  bool unpack_device; // true: device, false: cpu
  bool zeros_mode;    // true: fill inputs with 0.0f and skip golden matmul
};

struct DeviceParams {
  std::shared_ptr<tt::tt_metal::distributed::MeshDevice> device;
  CoreCoord grid_coord;
};

const char *input_dtype_cli_to_string(uint32_t input_dtype_cli);
const char *output_dtype_mode_to_string(OutputDTypeMode mode);
const char *effective_compute_dtype_string(uint32_t input_dtype_cli);

void log_validation_pipeline_steps(const TestParams &params) {
  uint32_t step = 1;
  log_info(LogTest, "Validation Pipeline (resolved steps)");
  log_info(LogTest, "----------------------------------");

  if (params.use_dram) {
    log_info(LogTest,
             "Step {}: Golden input policy = effective-input-golden (DRAM)",
             step++);
    log_info(LogTest,
             "Step {}: DType policy = input_cli={} (effective_compute_dtype={}), "
             "output_dtype={}",
             step++, input_dtype_cli_to_string(params.input_dtype),
             effective_compute_dtype_string(params.input_dtype),
             output_dtype_mode_to_string(params.output_dtype_mode));

    if (params.pack_device) {
      log_info(LogTest,
               "Step {}: Input prep = on-device ttnn tilize/pack for IN0/IN1",
               step++);
      log_info(LogTest,
               "Step {}: Reconstruct effective IN0/IN1 from packed DRAM buffers "
               "for golden-reference alignment",
               step++);
    } else {
      log_info(LogTest,
               "Step {}: Input prep = host tilize/pack for IN0/IN1 with "
               "effective-input decode for golden-reference alignment",
               step++);
    }

    if (params.unpack_device) {
      if (params.output_dtype_mode == OutputDTypeMode::Fp32) {
        log_info(LogTest,
                 "Step {}: Output readback = device unpack path "
                 "(typecast(TILE)->Finish->untilize->to_vector)",
                 step++);
      } else {
        log_info(LogTest,
                 "Step {}: Output readback = device unpack path "
                 "(untilize->to_vector, no explicit FP32 cast)",
                 step++);
      }
    } else {
      log_info(LogTest,
               "Step {}: Output readback = host unpack path "
               "(ReadShard packed -> host decode/untilize)",
               step++);
    }
  } else {
    log_info(LogTest,
             "Step {}: L1 mode uses per-core slicing/identity IN1 path; "
             "validation is best-effort and may not be academically rigorous",
             step++);
    log_info(LogTest,
             "Step {}: DType policy = input_cli={} (effective_compute_dtype={}), "
             "output_dtype={}",
             step++, input_dtype_cli_to_string(params.input_dtype),
             effective_compute_dtype_string(params.input_dtype),
             output_dtype_mode_to_string(params.output_dtype_mode));
    log_info(LogTest,
             "Step {}: Output readback = per-core L1 read + host decode/stitch",
             step++);
  }

  log_info(LogTest,
           "Step {}: Metrics = PCC + RMSE + Relative RMSE on aligned vectors",
           step++);
}

const char *test_type_to_string(TestType test) {
  switch (test) {
  case TestType::EmptyKernelLaunch:
    return "EmptyKernelLaunch";
  case TestType::ComputeMM:
    return "ComputeMM";
  case TestType::SubDeviceMM:
    return "SubDeviceMM";
  case TestType::HostPipelineComputeMM:
    return "HostPipelineComputeMM";
  case TestType::HostPipelineEmpty:
    return "HostPipelineEmpty";
  case TestType::ComputeMMAsyncBatch:
    return "ComputeMMAsyncBatch";
  case TestType::ComputeMMTraceReplay:
    return "ComputeMMTraceReplay";
  case TestType::InvalidTest:
    return "InvalidTest";
  default:
    return "Unknown";
  }
}

const char *arch_to_string(tt::ARCH arch) {
  switch (arch) {
  case tt::ARCH::GRAYSKULL:
    return "GRAYSKULL";
  case tt::ARCH::WORMHOLE_B0:
    return "WORMHOLE_B0";
  case tt::ARCH::BLACKHOLE:
    return "BLACKHOLE";
  default:
    return "UNKNOWN";
  }
}

const char *input_dtype_cli_to_string(uint32_t input_dtype_cli) {
  switch (input_dtype_cli) {
  case 0:
    return "BFP8";
  case 1:
    return "BF16";
  case 2:
    return "FP32";
  default:
    return "UNKNOWN";
  }
}

OutputDTypeMode parse_output_dtype_mode(const std::string &output_dtype_str) {
  if (output_dtype_str == "native") {
    return OutputDTypeMode::Native;
  }
  if (output_dtype_str == "bf16") {
    return OutputDTypeMode::Bf16;
  }
  if (output_dtype_str == "fp32") {
    return OutputDTypeMode::Fp32;
  }
  throw std::runtime_error("Invalid --output-dtype value: " +
                           output_dtype_str +
                           ". Must be 'native', 'bf16', or 'fp32'");
}

const char *output_dtype_mode_to_string(OutputDTypeMode mode) {
  switch (mode) {
  case OutputDTypeMode::Native:
    return "native";
  case OutputDTypeMode::Bf16:
    return "bf16";
  case OutputDTypeMode::Fp32:
    return "fp32";
  default:
    return "unknown";
  }
}

tt::DataFormat compute_data_format_from_input_dtype(uint32_t input_dtype) {
  // Current kernel contract supports BFP8/BF16 CB formats in this benchmark.
  // Input FP32 is accepted as CLI intent but maps to BF16 compute format.
  return (input_dtype == 0) ? tt::DataFormat::Bfp8_b : tt::DataFormat::Float16_b;
}

const char *effective_compute_dtype_string(uint32_t input_dtype_cli) {
  return (compute_data_format_from_input_dtype(input_dtype_cli) ==
          tt::DataFormat::Bfp8_b)
             ? "BFP8_B"
             : "BF16";
}

void log_effective_configuration(const TestParams &params,
                                 const DeviceParams &device_params,
                                 uint32_t max_x, uint32_t max_y) {
  const char *pack_mode = params.pack_device ? "device" : "cpu";
  const char *unpack_mode = params.unpack_device ? "device" : "cpu";
  const char *input_dtype_cli = input_dtype_cli_to_string(params.input_dtype);
  const char *effective_dtype = effective_compute_dtype_string(params.input_dtype);
  const char *output_dtype = output_dtype_mode_to_string(params.output_dtype_mode);
  const char *validation_mode =
      (params.use_dram && !params.bypass_check) ? "effective-input-golden"
                                                 : "best-effort";

  log_info(LogTest, "Resolved Test Configuration");
  log_info(LogTest, "===========================");
  log_info(LogTest, "test={} ({})", static_cast<uint32_t>(params.test),
           test_type_to_string(params.test));
  log_info(LogTest, "arch={}, grid_max={}x{}",
           arch_to_string(device_params.device->arch()), max_x, max_y);
  log_info(LogTest,
           "M={}, N={}, K={}, dtype_cli={}, input_dtype_cli={}, "
           "effective_compute_dtype={}, output_dtype={}, "
           "fidel={}, dram={}, pack_tile={}, unpack_tile={}",
           params.M, params.N, params.K, params.dtype, input_dtype_cli,
           effective_dtype, output_dtype,
           params.fidel, params.use_dram ? "on" : "off", pack_mode,
           unpack_mode);
  log_info(LogTest,
           "core_x={}, core_y={}, core_groups={}, num_iters={}, "
           "num_rt_args={}, trace_capture_ops={}, bypass_check={}, cache={}, "
           "clean_mode={}, cpu_id={}, cpu_range={}",
           params.core_x, params.core_y, params.core_groups, params.num_iters,
           params.num_rt_args, params.trace_capture_ops,
           params.bypass_check ? "on" : "off",
           params.use_cache ? "on" : "off", params.clean_mode,
           params.cpu_id, params.cpu_range);
  log_info(LogTest, "validation_mode={}, input_mode={}", validation_mode,
           params.zeros_mode ? "zeros" : "random");

  if (params.input_dtype == 2) {
    log_warning(LogTest,
                "input_dtype_cli=FP32 currently maps to effective_compute_dtype=BF16 "
                "in this benchmark path.");
  }
}

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
  uint32_t trace_capture_ops;
  uint32_t cpu_id;
  uint32_t cpu_range;
  uint32_t clean_mode;
  TestType test;
  bool pack_device = false;
  bool unpack_device = false;
  uint32_t input_dtype = 0;
  OutputDTypeMode output_dtype_mode = OutputDTypeMode::Native;
  uint32_t test_uint;
  std::string input_dtype_str;
  std::string output_dtype_str;
  std::string pack_tile_str;
  std::string unpack_tile_str;
  std::string input_mode_str;
  bool zeros_mode = false;
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
    if (dtype > 2) {
      throw std::runtime_error("Invalid --dtype value: " +
                               std::to_string(dtype) +
                               ". Must be 0 (BFP8), 1 (BF16), or 2 (FP32)");
    }
    std::tie(fidel, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(input_args,
                                                                "--fidel", 0);

    // New split dtype controls (backward-compatible with --dtype)
    std::tie(input_dtype_str, input_args) =
        test_args::get_command_option_and_remaining_args(
            input_args, "--input-dtype", "inherit");
    if (input_dtype_str == "inherit") {
      input_dtype = dtype;
    } else if (input_dtype_str == "bfp8") {
      input_dtype = 0;
    } else if (input_dtype_str == "bf16") {
      input_dtype = 1;
    } else if (input_dtype_str == "fp32") {
      input_dtype = 2;
    } else {
      throw std::runtime_error("Invalid --input-dtype value: " +
                               input_dtype_str +
                               ". Must be 'bfp8', 'bf16', 'fp32', or 'inherit'");
    }

    std::tie(output_dtype_str, input_args) =
        test_args::get_command_option_and_remaining_args(
            input_args, "--output-dtype", "native");
    output_dtype_mode = parse_output_dtype_mode(output_dtype_str);

    std::tie(use_dram, input_args) =
        test_args::has_command_option_and_remaining_args(input_args, "--dram");
    std::tie(use_cache, input_args) =
        test_args::has_command_option_and_remaining_args(input_args, "--cache");

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
    std::tie(trace_capture_ops, input_args) =
      test_args::get_command_option_uint32_and_remaining_args(
        input_args, "--trace-capture-ops", 1);
    std::tie(cpu_id, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(
            input_args, "--cpu", 0xFFFFFFFF);
    std::tie(cpu_range, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(
            input_args, "--cpu-range", 4);

    std::tie(clean_mode, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(
            input_args, "--clean-mode", 0);

    std::tie(test_uint, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(input_args,
                    "--test", 7);

    std::tie(unpack_tile_str, input_args) =
        test_args::get_command_option_and_remaining_args(
            input_args, "--unpack-tile", "cpu");
    if (unpack_tile_str == "device") {
      unpack_device = true;
    } else if (unpack_tile_str == "cpu") {
      unpack_device = false;
    } else {
      throw std::runtime_error("Invalid --unpack-tile value: " +
                               unpack_tile_str + ". Must be 'cpu' or 'device'");
    }

    // Backward-compatible default: if --pack-tile is not provided,
    // it inherits the value from --unpack-tile.
    std::tie(pack_tile_str, input_args) =
        test_args::get_command_option_and_remaining_args(
            input_args, "--pack-tile", "inherit");
    if (pack_tile_str == "inherit") {
      pack_device = unpack_device;
    } else if (pack_tile_str == "device") {
      pack_device = true;
    } else if (pack_tile_str == "cpu") {
      pack_device = false;
    } else {
      throw std::runtime_error("Invalid --pack-tile value: " +
                               pack_tile_str +
                               ". Must be 'cpu', 'device', or 'inherit'");
    }

    std::tie(input_mode_str, input_args) =
        test_args::get_command_option_and_remaining_args(
            input_args, "--input-mode", "random");
    if (input_mode_str == "random") {
      zeros_mode = false;
    } else if (input_mode_str == "zeros") {
      zeros_mode = true;
    } else {
      throw std::runtime_error("Invalid --input-mode value: " + input_mode_str +
                               ". Must be 'random' or 'zeros'");
    }

    if (test_uint > static_cast<uint32_t>(TestType::InvalidTest)) {
      throw std::runtime_error("Invalid --test value");
    }
    if (trace_capture_ops == 0) {
      throw std::runtime_error(
          "Invalid --trace-capture-ops value: must be >= 1");
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
                    .input_dtype = input_dtype,
                    .output_dtype_mode = output_dtype_mode,
                    .fidel = fidel,
                    .use_dram = use_dram,
                    .use_cache = use_cache,
                    .core_x = core_x,
                    .core_y = core_y,
                    .core_groups = core_groups,
                    .num_iters = num_iters,
                    .bypass_check = bypass_check,
                    .num_rt_args = num_rt_args,
                    .trace_capture_ops = trace_capture_ops,
                    .cpu_id = cpu_id,
                    .cpu_range = cpu_range,
                    .clean_mode = clean_mode,
                    .test = test,
                    .pack_device = pack_device,
                    .unpack_device = unpack_device,
                    .zeros_mode = zeros_mode};
  return params;
}

///////////////////////////////////////////////////////////////////////////////////////////
/// Compute MM Test Helpers
///
/// These functions help calculate the optimal blocking parameters for the
/// specific architecture and memory constraints.
///////////////////////////////////////////////////////////////////////////////////////////

// Returns the L1 memory size available per core for the specific architecture
// 1-to-1 match with 1_compute_mm/test_compute_mm.cpp
uint32_t get_l1_size(tt::ARCH arch) {
  constexpr uint32_t GS_L1_SIZE = 1048576;
  constexpr uint32_t WH_L1_SIZE = 1499136;
  constexpr uint32_t BH_L1_SIZE = 1499136;

  uint32_t l1_size = 0;
  if (arch == tt::ARCH::WORMHOLE_B0) {
    l1_size = WH_L1_SIZE;
  } else if (arch == tt::ARCH::GRAYSKULL) {
    l1_size = GS_L1_SIZE;
  } else if (arch == tt::ARCH::BLACKHOLE) {
    l1_size = BH_L1_SIZE;
  }
  return l1_size;
}

// Aligns the Matrix dimensions (M, N, K) to the tile size (32x32).
// Returns the number of tiles in each dimension (Mt, Nt, Kt).
// e.g., if M=32, Mt=1. If M=64, Mt=2.
// 1-to-1 match with 1_compute_mm/test_compute_mm.cpp
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

// Determines the optimal 'block width' (in number of tiles) for the input
// matrix A (in0). The block width determines how many tiles along the K
// dimension are loaded into L1 at once. This function iterates through choices
// (4, 2, 1) and picks the largest one that fits in L1 memory alongside other
// required buffers. 1-to-1 match with 1_compute_mm/test_compute_mm.cpp
uint32_t get_in0_block_w(uint32_t per_core_Mt, uint32_t per_core_Nt,
                         uint32_t Kt, uint32_t single_tile_size,
                         uint32_t l1_size, uint32_t l1_unreserved_base,
                         bool use_dram = false) {
  std::vector<uint32_t> in0_block_w_choices = {4, 2, 1};
  uint32_t num_buffer = 2; // double buffering
  uint32_t in0_block_w = 0;
  uint32_t base_addr = l1_unreserved_base;
  for (auto choice : in0_block_w_choices) {
    if (Kt % choice != 0) {
      continue;
    }

    uint32_t in0_cb_size = per_core_Mt * choice * num_buffer * single_tile_size;
    uint32_t in1_cb_size = per_core_Nt * choice * num_buffer * single_tile_size;
    uint32_t in2_cb_size = 2 * single_tile_size;
    uint32_t intermediate_cb_size =
        per_core_Mt * per_core_Nt * single_tile_size;
    uint32_t out_cb_size = per_core_Mt * per_core_Nt * single_tile_size;

    uint32_t total_cb_size = in0_cb_size + in1_cb_size + in2_cb_size +
                             intermediate_cb_size + out_cb_size;

    // In DRAM mode, data tensors live in DRAM, not L1.
    // Only circular buffers need L1 space.
    uint32_t total_l1_needed = total_cb_size;
    if (!use_dram) {
      uint32_t per_core_in0_size = per_core_Mt * choice * single_tile_size;
      uint32_t per_core_in1_size = per_core_Nt * choice * single_tile_size;
      uint32_t per_core_out_size = per_core_Mt * per_core_Nt * single_tile_size;
      total_l1_needed +=
          per_core_in0_size + per_core_in1_size + per_core_out_size;
    }

    if (base_addr + total_l1_needed <= l1_size) {
      in0_block_w = choice;
      break;
    }
  }
  return in0_block_w;
}

// ============================================================================
// L1 Heuristic Block Allocator (DRAM Mode)
// Solves for optimal `out_block_h`, `out_block_w` and `in0_block_w` config
// such that Circular Buffers fit into the core's L1 SRAM.
// Based heavily on TTNN matmul_program_config formulas.
// ============================================================================
std::tuple<uint32_t, uint32_t, uint32_t>
get_dynamic_l1_block_params(uint32_t per_core_Mt, uint32_t per_core_Nt,
                            uint32_t Kt, uint32_t single_tile_size,
                            uint32_t l1_size, uint32_t l1_unreserved_base) {

  // Safe watermark: total L1 space available for Circular Buffers
  // TTNN's dynamic heuristic uses exactly l1_size - l1_unreserved_base
  uint32_t max_l1_usage = l1_size - l1_unreserved_base;

  std::vector<uint32_t> in0_block_w_choices = {4, 2, 1};
  std::vector<std::tuple<uint32_t, uint32_t>> dim_divisors;

  // Find all possible integer divisions of the core's grid size
  for (uint32_t h_div = 1; h_div <= per_core_Mt; ++h_div) {
    if (per_core_Mt % h_div != 0)
      continue;
    for (uint32_t w_div = 1; w_div <= per_core_Nt; ++w_div) {
      if (per_core_Nt % w_div != 0)
        continue;
      dim_divisors.push_back({per_core_Mt / h_div, per_core_Nt / w_div});
    }
  }

  // Iterate from largest output block (whole core) to smallest
  for (const auto &[out_bh, out_bw] : dim_divisors) {
    for (uint32_t in0_bw : in0_block_w_choices) {
      if (Kt % in0_bw != 0)
        continue;

      // Memory footprint formulas (Deepwiki Section 2)
      uint32_t cb_in0 = out_bh * in0_bw * 2 * single_tile_size; // double
      uint32_t cb_in1 = out_bw * in0_bw * 2 * single_tile_size; // double
      uint32_t cb_interm =
          out_bh * out_bw * single_tile_size; // shared with cb_out
      uint32_t cb_in2 =
          2 * single_tile_size; // 2 zero pad tiles (TTNN alignment)

      uint32_t total_cb = cb_in0 + cb_in1 + cb_interm + cb_in2;

      if (total_cb <= max_l1_usage) {
        return {out_bh, out_bw, in0_bw};
      }
    }
  }

  // Fallback (will trigger L1 crash logging downstream)
  return {per_core_Mt, per_core_Nt, 0};
}

////////////////////////////////////////////////////////////////////////////////
// Helper Struct for Benchmark I/O
//
// This struct bundles all data returned by prepare_inputs_compute_mm().
// It stores the original FP32 input vectors (needed by the host-side golden
// reference validation) and, when DRAM mode is active, shared pointers to
// the device-side DRAM interleaved buffers.
//
// WHY?
//   - L1 mode:  Data is written directly to each core's L1 SRAM using
//               WriteToDeviceL1(). The kernel reads from a fixed local
//               address. No Buffer objects are needed.
//   - DRAM mode: Data is stored in device DRAM using tt_metal::CreateBuffer()
//                with InterleavedBufferConfig. Tiles are distributed across
//                DRAM banks in round-robin order (page 0 → bank 0,
//                page 1 → bank 1, ...). The kernel uses
//                InterleavedAddrGenFast<true> to compute NoC addresses at
//                runtime given the buffer's base address and page size.
//                We need the Buffer objects here so we can:
//                  1. Extract buffer->address() for runtime args (see
//                     test_compute_mm)
//                  2. Call ReadFromBuffer() for host-side verification
//                  3. Keep the buffers alive (shared_ptr prevents dealloc)
////////////////////////////////////////////////////////////////////////////////
struct BenchmarkInputs {
  // Original FP32 input matrices — kept for golden-reference comparison.
  // in0_vec: Matrix A, row-major, size = Mt*32 * Kt*32
  // in1_vec: Matrix B, row-major, size = Kt*32 * Nt*32
  std::vector<float> in0_vec;
  std::vector<float> in1_vec;

  // Device-side DRAM interleaved buffers (MeshBuffer for distributed API).
  // These are nullptr when use_dram=false (L1 mode).
  // When use_dram=true, each buffer holds tilized+packed data distributed
  // across all DRAM banks via MeshBuffer::create(ReplicatedBufferConfig).
  std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>
      in0_buffer; // IN0 (A matrix) in DRAM
  std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>
      in1_buffer; // IN1 (B matrix) in DRAM
  std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>
      out_buffer; // Output (C matrix) in DRAM

  // Dedicated handle for output device tensor when device-side unpack/readback
  // is requested. This avoids relying on positional assumptions in
  // ttnn_tensors.
  std::optional<ttnn::Tensor> output_ttnn_tensor;

  // Keep ttnn tensors alive if used for on-device tilization
  std::vector<ttnn::Tensor> ttnn_tensors;
};

// ============================================================================
// Forward Declarations
// ============================================================================

// Creates the device program (circular buffers, kernels, runtime args) for
// matrix multiplication C = A * B.  When use_dram=true, DRAM-specific kernels
// are compiled and runtime args use global matrix strides (Kt, Nt).
// When use_dram=false, the original L1 kernels from 1_compute_mm are used.
//
// Parameters:
//   device           - mesh device handle
//   cb_data_format   - tile data format (Bfp8_b or Float16_b)
//   math_fidelity    - compute fidelity (LoFi / HiFi)
//   fp32_dest_acc_en - enable FP32 accumulation in DST registers
//   single_tile_size - size in bytes of one tile (depends on data format)
//   core_range       - (num_cores_x, num_cores_y) grid dimensions
//   Mt, Nt, Kt       - matrix dims in tiles (M/32, N/32, K/32)
//   in0_block_w      - block width along K dimension (tiles per block)
//   out_subblock_h/w - subblock dims for register accumulation
//   per_core_Mt/Nt   - tiles per core in M and N dimensions
//   in0/1/2_cb_addr  - L1 circular buffer base addresses
//   out_cb_addr      - L1 output circular buffer base address
//   in0/1_addr       - L1 or DRAM base address for IN0/IN1 data
//   out_addr         - L1 or DRAM base address for output data
//   use_dram         - if true, select DRAM kernels and global strides
//   program          - reference to the Program object (kernels appended)
//   start_core_y     - first Y-coordinate (for sub-device splits)
void create_program_compute_mm(
    tt_metal::distributed::MeshDevice *device, tt::DataFormat cb_data_format,
    MathFidelity math_fidelity, bool fp32_dest_acc_en,
    uint32_t single_tile_size, CoreCoord core_range, uint32_t Mt, uint32_t Nt,
    uint32_t Kt, uint32_t in0_block_w, uint32_t out_subblock_h,
    uint32_t out_subblock_w, uint32_t per_core_Mt, uint32_t per_core_Nt,
    uint32_t out_block_h, uint32_t out_block_w, uint32_t num_blocks_h,
    uint32_t num_blocks_w, uint32_t in0_cb_addr, uint32_t in1_cb_addr,
    uint32_t in2_cb_addr, uint32_t out_cb_addr, uint32_t interm_cb_addr,
    uint32_t in0_addr, uint32_t in1_addr, uint32_t out_addr, bool use_dram,
    tt_metal::Program &program, uint32_t start_core_y = 0);

// Generates random FP32 input matrices, tilizes/packs them, and transfers
// them to the device. Returns a BenchmarkInputs struct.
//
// L1 mode:  Slices the matrix per-core, tilizes each slice, and writes
//           to each core's L1 at in0_addr / in1_addr.
// DRAM mode: Tilizes the FULL matrix, packs into BFP8, creates DRAM
//            interleaved buffers (one contiguous buffer per matrix),
//            and writes via WriteToBuffer(). The output DRAM buffer is
//            also created (empty) to receive kernel results.
BenchmarkInputs prepare_inputs_compute_mm(
    tt_metal::distributed::MeshDevice *device, CoreCoord core_range,
    uint32_t Mt, uint32_t Nt, uint32_t Kt, uint32_t per_core_Mt,
    uint32_t per_core_Nt, uint32_t in0_block_w, uint32_t single_tile_size,
    uint32_t in0_addr, uint32_t in1_addr, uint32_t in2_cb_addr, bool use_dram,
    uint32_t start_core_y = 0,
    tt::DataFormat data_format = tt::DataFormat::Bfp8_b,
    bool pack_device = false,
  bool reconstruct_effective_inputs = false,
  bool create_output_ttnn_tensor = false,
  bool zeros_mode = false);

std::tuple<MathFidelity, bool> get_compute_params(tt::ARCH arch);

std::tuple<uint32_t, uint32_t> get_out_subblock_params(uint32_t per_core_Mt,
                                                       uint32_t per_core_Nt,
                                                       uint32_t choice = 0);

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
           uint32_t>
get_all_buffers_addresses(uint32_t per_core_Mt, uint32_t per_core_Nt,
                          uint32_t in0_block_w, uint32_t single_tile_size,
                          uint32_t l1_unreserved_base, bool use_dram = false);

//! Phase 2: Sub-Device Parallelism Test
//! Partitions the device into 2 Sub-Devices and runs independent MM workloads
//! on each.
bool test_sub_device_manager_mm(tt_metal::distributed::MeshDevice *device,
                                const TestParams &params) {
  bool pass = true;
  try {
    log_info(LogTest, "Starting Phase 2: Sub-Device Parallelism Test");

    if (params.core_y < 2) {
      log_fatal(
          LogTest,
          "Sub-Device test requires at least 2 rows of cores (y_size >= 2)");
      return false;
    }

    uint32_t num_sub_devices = params.core_groups;
    if (num_sub_devices < 2) {
      log_info(
          LogTest,
          "Note: SubDevice test running with 1 Core Group (Single SubDevice)");
      num_sub_devices = 1;
    }

    uint32_t rows_per_sub_device = params.core_y / num_sub_devices;

    // Test 2 DRAM compromise policy (isolated to this test only):
    // keep split math and runtime-arg generation simple/safe by requiring
    // equal M partitioning across sub-devices.
    if (params.use_dram && (params.M % num_sub_devices != 0)) {
      log_error(LogTest,
                "Test 2 DRAM currently requires M ({}) divisible by "
                "core_groups ({}).",
                params.M, num_sub_devices);
      return false;
    }

    if (params.use_dram && params.pack_device) {
      log_warning(LogTest,
                  "Test 2 DRAM: --pack-tile device is currently not enabled; "
                  "falling back to CPU pack for this test.");
    }

    if (params.unpack_device ||
        params.output_dtype_mode != OutputDTypeMode::Native) {
      log_warning(LogTest,
                  "Test 2 does not perform output readback/export. "
                  "--unpack-tile/--output-dtype are ignored for this test.");
    }

    const bool test2_pack_device = params.use_dram ? false : params.pack_device;

    // 1. Get Common Params
    auto arch = device->arch();
    uint32_t l1_size = get_l1_size(arch);
    uint32_t l1_unreserved_base =
        device->allocator()->get_base_allocator_addr(HalMemType::L1);
    auto [math_fidelity, fp32_dest_acc_en] = get_compute_params(arch);
    tt::DataFormat data_format =
      compute_data_format_from_input_dtype(params.input_dtype);
    uint32_t single_tile_size = tt::tile_size(data_format);

    // 2. Prepare Program with Loop over Splits
    tt_metal::Program program;
    uint32_t M_split = params.M / num_sub_devices;
    std::vector<BenchmarkInputs> split_inputs;
    split_inputs.reserve(num_sub_devices);

    for (uint32_t i = 0; i < num_sub_devices; i++) {
      // Define Core Range for this Split
      uint32_t start_y = i * rows_per_sub_device;
      uint32_t end_y = (i == num_sub_devices - 1)
                           ? params.core_y - 1
                           : (start_y + rows_per_sub_device - 1);
      CoreCoord split_grid_size = {(std::size_t)params.core_x,
                                   (std::size_t)(end_y - start_y + 1)};

      log_info(LogTest, "SubDevice {}: Rows {}-{} (M={})", i, start_y, end_y,
               M_split);

      // Calculate Blockings for this Split
      auto [Mt, Nt, Kt] =
          get_aligned_input_tile_num(M_split, params.N, params.K);

      uint32_t per_core_Mt = ((Mt - 1) / split_grid_size.y) + 1;
      uint32_t per_core_Nt = ((Nt - 1) / split_grid_size.x) + 1;

      uint32_t out_block_h = per_core_Mt;
      uint32_t out_block_w = per_core_Nt;
      uint32_t num_blocks_h = 1;
      uint32_t num_blocks_w = 1;

      uint32_t in0_block_w =
          get_in0_block_w(per_core_Mt, per_core_Nt, Kt, single_tile_size,
                          l1_size, l1_unreserved_base, params.use_dram);
      if (in0_block_w == 0)
        throw std::runtime_error("Insufficient L1 for SubDevice split");

      if (params.use_dram) {
        auto [dram_out_block_h, dram_out_block_w, dram_in0_block_w] =
            get_dynamic_l1_block_params(per_core_Mt, per_core_Nt, Kt,
                                        single_tile_size, l1_size,
                                        l1_unreserved_base);
        out_block_h = dram_out_block_h;
        out_block_w = dram_out_block_w;
        in0_block_w = dram_in0_block_w;
        num_blocks_h = per_core_Mt / out_block_h;
        num_blocks_w = per_core_Nt / out_block_w;
      }

      auto [out_subblock_h, out_subblock_w] =
          get_out_subblock_params(out_block_h, out_block_w);

      auto [in0_cb_addr, in1_cb_addr, in2_cb_addr, out_cb_addr, interm_cb_addr,
            in0_addr, in1_addr, out_addr] =
          get_all_buffers_addresses(out_block_h, out_block_w, in0_block_w,
                                    single_tile_size, l1_unreserved_base,
                                    params.use_dram);

      // Prepare Inputs for this split
      auto inputs_split = prepare_inputs_compute_mm(
          device, split_grid_size, Mt, Nt, Kt, per_core_Mt, per_core_Nt,
          in0_block_w, single_tile_size, in0_addr, in1_addr, in2_cb_addr,
          params.use_dram, start_y, data_format, test2_pack_device,
          /*reconstruct_effective_inputs=*/false,
          /*create_output_ttnn_tensor=*/false,
          /*zeros_mode=*/params.zeros_mode);

      uint32_t effective_in0_addr = in0_addr;
      uint32_t effective_in1_addr = in1_addr;
      uint32_t effective_out_addr = out_addr;
      if (params.use_dram) {
        auto coord = tt::tt_metal::distributed::MeshCoordinate(0, 0);
        effective_in0_addr =
            inputs_split.in0_buffer->get_device_buffer(coord)->address();
        effective_in1_addr =
            inputs_split.in1_buffer->get_device_buffer(coord)->address();
        effective_out_addr =
            inputs_split.out_buffer->get_device_buffer(coord)->address();
      }

      // Create kernels/buffers for this split after addresses are resolved.
      create_program_compute_mm(
          device, data_format, math_fidelity, fp32_dest_acc_en,
          single_tile_size, split_grid_size, Mt, Nt, Kt, in0_block_w,
          out_subblock_h, out_subblock_w, per_core_Mt, per_core_Nt, out_block_h,
          out_block_w, num_blocks_h, num_blocks_w, in0_cb_addr, in1_cb_addr,
          in2_cb_addr, out_cb_addr, interm_cb_addr, effective_in0_addr,
          effective_in1_addr, effective_out_addr, params.use_dram, program,
          start_y);

      // Keep per-split buffers/tensors alive across the whole dispatch loop.
      split_inputs.push_back(std::move(inputs_split));
    }

    // 3. Profiling Loop (Standard Dispatch)
    auto mesh_workload = tt_metal::distributed::MeshWorkload();
    mesh_workload.add_program(
        tt::tt_metal::distributed::MeshCoordinateRange{{0, 0}, {0, 0}},
        std::move(program));

    log_info(LogTest, "Num tests {}", params.num_iters);
    for (uint32_t i = 0; i < params.num_iters; ++i) {
      ZoneScopedN("Sub-Device Parallel Dispatch");
      ZoneValue(i);
      tt_metal::distributed::EnqueueMeshWorkload(device->mesh_command_queue(),
                                                 mesh_workload, false);
      tt_metal::distributed::Finish(device->mesh_command_queue());
    }

    return pass;
  } catch (const std::exception &e) {
    log_error(LogTest, "Error: {}", e.what());
    return false;
  }
}

bool test_host_pipeline_compute_mm(tt_metal::distributed::MeshDevice *device,
                                   const TestParams &params);

bool test_host_pipeline_empty_tensor(tt_metal::distributed::MeshDevice *device,
                                     const TestParams &params);

bool test_compute_mm_async(tt_metal::distributed::MeshDevice *device,
                           const TestParams &params);

bool test_compute_mm_trace(tt_metal::distributed::MeshDevice *device,
                           const TestParams &params);

// 1-to-1 match with 1_compute_mm/test_compute_mm.cpp
// MODIFIED: FP32 ACCUM TO TRUE -> poor preceision
std::tuple<MathFidelity, bool> get_compute_params(tt::ARCH arch) {
  MathFidelity math_fidelity = MathFidelity::HiFi4;
  bool fp32_dest_acc_en = false;
  if (arch == tt::ARCH::WORMHOLE_B0 or arch == tt::ARCH::BLACKHOLE) {
    math_fidelity = MathFidelity::HiFi2;
    fp32_dest_acc_en = false;
  } else if (arch == tt::ARCH::GRAYSKULL) {
    math_fidelity = MathFidelity::HiFi4;
    fp32_dest_acc_en = false;
  }
  return {math_fidelity, fp32_dest_acc_en};
}

// Determines optimal subblock dimensions for the compute kernel.
// Subblocks are smaller units of work within a larger block that fit into the
// compute engine's registers (DST register). The goal is to maximize register
// reuse. Typical sizes are 4x2, 2x4, etc. 1-to-1 match with
// 1_compute_mm/test_compute_mm.cpp
std::tuple<uint32_t, uint32_t> get_out_subblock_params(uint32_t per_core_Mt,
                                                       uint32_t per_core_Nt,
                                                       uint32_t choice) {
  constexpr std::array<std::tuple<uint32_t, uint32_t>, 20> SUBBLOCK_HW_CHOICES =
      {{
          {4, 2}, {2, 4}, {8, 1}, {1, 8}, {7, 1}, {1, 7}, {3, 2},
          {2, 3}, {6, 1}, {1, 6}, {5, 1}, {1, 5}, {2, 2}, {4, 1},
          {1, 4}, {3, 1}, {1, 3}, {2, 1}, {1, 2}, {1, 1},
      }};

  uint32_t index = 0;
  for (const auto &subblock_hw : SUBBLOCK_HW_CHOICES) {
    auto subblock_h = std::get<0>(subblock_hw);
    auto subblock_w = std::get<1>(subblock_hw);
    if (per_core_Mt % subblock_h == 0 and per_core_Nt % subblock_w == 0) {
      if (index >= choice) {
        return {subblock_h, subblock_w};
      }
      index++;
    }
  }

  return {1, 1};
}

// Calculates the precise L1 addresses for all circular buffers (CBs) and data
// tensors. In TT-Metal, specific L1 regions are allocated for "Circular
// Buffers" (input/output/intermediate data) and "Tensors" (if resident in L1).
// This function assumes a contiguous layout starting from `l1_unreserved_base`.
// Structure: [IN0_CB][IN1_CB][IN2_CB][OUT_CB][...Free Space...]
// 1-to-1 match with 1_compute_mm/test_compute_mm.cpp
std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
           uint32_t>
get_all_buffers_addresses(uint32_t per_core_Mt, uint32_t per_core_Nt,
                          uint32_t in0_block_w, uint32_t single_tile_size,
                          uint32_t l1_unreserved_base, bool use_dram) {
  uint32_t num_buffer = 2; // double buffering
  uint32_t in0_cb_addr = l1_unreserved_base;
  uint32_t in0_cb_size =
      per_core_Mt * in0_block_w * num_buffer * single_tile_size;
  uint32_t in1_cb_addr = in0_cb_addr + in0_cb_size;
  uint32_t in1_cb_size =
      per_core_Nt * in0_block_w * num_buffer * single_tile_size;
  uint32_t in2_cb_addr = in1_cb_addr + in1_cb_size;
  uint32_t in2_cb_size = 2 * single_tile_size;
  uint32_t interm_cb_addr = in2_cb_addr + in2_cb_size;
  uint32_t interm_cb_size = per_core_Mt * per_core_Nt * single_tile_size;
  uint32_t out_cb_addr = interm_cb_addr + interm_cb_size;
  uint32_t out_cb_size = per_core_Mt * per_core_Nt * single_tile_size;

  // In DRAM mode, data tensors live in DRAM — don't allocate L1 for them.
  // Addresses will be overridden by DRAM buffer addresses in test_compute_mm.
  uint32_t in0_addr = 0;
  uint32_t in1_addr = 0;
  uint32_t out_addr = 0;
  if (!use_dram) {
    uint32_t per_core_in0_tiles = per_core_Mt * in0_block_w;
    uint32_t per_core_in1_tiles = per_core_Nt * in0_block_w;
    in0_addr = out_cb_addr + out_cb_size;
    in1_addr = in0_addr + (per_core_in0_tiles * single_tile_size);
    out_addr = in1_addr + (per_core_in1_tiles * single_tile_size);
  }

  return {in0_cb_addr,    in1_cb_addr, in2_cb_addr, out_cb_addr,
          interm_cb_addr, in0_addr,    in1_addr,    out_addr};
}

// MODIFIED: Added Tracy ZoneScopedN for profiling.
// Originally from 1_compute_mm/test_compute_mm.cpp
// Originally from 1_compute_mm/test_compute_mm.cpp
std::vector<float> generate_fp32_random(uint32_t num_elems,
                                        float scale = 1.0f) {
  ZoneScopedN("Generate FP32 Random");
  std::vector<float> vec(num_elems);
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  // Generate random numbers in range [-scale, scale]
  // This mimics He/Xavier initialization when scale = 1/sqrt(K)
  auto rand_float = std::bind(
      std::uniform_real_distribution<float>(-scale, scale), std::mt19937(seed));
  for (uint32_t i = 0; i < num_elems; ++i) {
    vec.at(i) = rand_float();
  }
  return vec;
}

// 1-to-1 match with 1_compute_mm/test_compute_mm.cpp
template <typename T>
std::vector<T> get_row_slice(std::vector<T> data, int start_row_index,
                             int num_rows, int /*rows*/, int cols) {
  std::vector<T> result;
  for (int i = start_row_index * cols; i < (start_row_index + num_rows) * cols;
       i++) {
    result.push_back(data.at(i));
  }
  return result;
}

// 1-to-1 match with 1_compute_mm/test_compute_mm.cpp
template <typename T>
std::vector<T> get_col_slice(std::vector<T> data, int start_col_index,
                             int num_cols, int rows, int cols) {
  std::vector<T> result;
  for (int r = 0; r < rows; r++) {
    for (int c = start_col_index; c < (start_col_index + num_cols); c++) {
      result.push_back(data.at((r * cols) + c));
    }
  }
  return result;
}

// Calculates the Pearson Correlation Coefficient (PCC) between two vectors.
// Used to validate BFP8/FP16 outputs against FP32 golden reference.
// Calculates the Pearson Correlation Coefficient (PCC) between two vectors.
// Uses a numerically stable 2-pass algorithm (centered) to avoid catastrophic
// cancellation.
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

// Calculates Root Mean Square Error (RMSE)
// Calculates Root Mean Square Error (RMSE)
float get_rmse(const std::vector<float> &x, const std::vector<float> &y) {
  if (x.size() != y.size()) {
    return std::numeric_limits<float>::infinity();
  }

  double sum_sq_diff = 0.0;
  for (size_t i = 0; i < x.size(); ++i) {
    double diff = static_cast<double>(x[i]) - static_cast<double>(y[i]);
    sum_sq_diff += diff * diff;
  }
  return static_cast<float>(std::sqrt(sum_sq_diff / x.size()));
}

// Calculates Relative RMSE: ||x - ref||_2 / ||ref||_2
// Implementation provided by User for academic rigor.
float get_relative_rmse(const std::vector<float> &x,
                        const std::vector<float> &ref) {
  if (x.size() != ref.size()) {
    return std::numeric_limits<float>::quiet_NaN();
  }

  const size_t n = x.size();
  double err_sq_sum = 0.0;
  double ref_sq_sum = 0.0;

  for (size_t i = 0; i < n; ++i) {
    double diff = static_cast<double>(x[i]) - ref[i];
    err_sq_sum += diff * diff;
    ref_sq_sum += static_cast<double>(ref[i]) * ref[i];
  }

  if (ref_sq_sum == 0.0) {
    return std::numeric_limits<float>::quiet_NaN(); // undefined
  }

  double rrmse = std::sqrt(err_sq_sum / ref_sq_sum);
  return static_cast<float>(rrmse);
}

// Maximum absolute value in a vector — used for zeros-mode validation
// (golden = all zeros, so a passing run has max-abs == 0 within tolerance).
float get_max_abs(const std::vector<float> &v) {
  float m = 0.0f;
  for (float x : v) {
    float a = std::fabs(x);
    if (a > m) m = a;
  }
  return m;
}

// Reference Matrix Multiplication (row-major)
// C = A * B. A is (M x K), B is (K x N), C is (M x N)
// Uses double precision for accumulation to serve as a rigorous Golden
// Reference.
std::vector<float> matmul_reference(const std::vector<float> &a,
                                    const std::vector<float> &b, uint32_t M,
                                    uint32_t N, uint32_t K) {
  std::vector<float> c(M * N, 0.0f);
  for (uint32_t i = 0; i < M; ++i) {
    for (uint32_t k = 0; k < K; ++k) {
      double val_a = static_cast<double>(a[i * K + k]);
      for (uint32_t j = 0; j < N; ++j) {
        // Accumulate in double to minimize rounding errors
        double current_val = static_cast<double>(c[i * N + j]);
        current_val += val_a * static_cast<double>(b[k * N + j]);
        c[i * N + j] = static_cast<float>(current_val);
      }
    }
  }
  return c;
}

////////////////////////////////////////////////////////////////////////////////
// prepare_inputs_compute_mm
//
// Generates random FP32 input matrices A (MxK) and B (KxN), converts them to
// tilized BFP8 format, and transfers them to the device. The function handles
// two distinct data paths:
//
// L1 MODE (use_dram=false):
//   For each core (y,x), the function:
//     1. Slices out the core's row-block from IN0 (get_row_slice →
//     get_col_slice)
//     2. Tilizes the slice (32x32 tile groups) and packs to BFP8
//     3. Writes the tile block to the core's L1 at `in0_addr`
//     4. Generates an identity-like IN1 block for that core and writes to
//     `in1_addr`
//     5. Writes a zeros tile to `in2_cb_addr` (used as bias/padding by compute
//     kernel)
//   This is the original approach from 1_compute_mm/test_compute_mm.cpp.
//   The kernels read from local L1 addresses using noc_async_read() with the
//   core's own NOC coordinates (phy_core.x, phy_core.y).
//
// DRAM MODE (use_dram=true):
//   Instead of splitting per-core, the ENTIRE matrix is tilized at once:
//     1. Tilizes the full IN0 (Mt*32 × Kt*32) and packs to BFP8
//     2. Creates a DRAM interleaved buffer via
//     CreateBuffer(InterleavedBufferConfig)
//        - Tiles are distributed round-robin across DRAM banks (page 0 → bank
//        0,
//          page 1 → bank 1, ..., page N → bank 0, etc.)
//        - The hardware has multiple DRAM channels; interleaving maximizes
//        bandwidth
//     3. Writes the packed tile data to the buffer via WriteToBuffer()
//     4. Repeats for IN1 (Kt*32 × Nt*32) and creates an empty output buffer
//     (Mt*Nt tiles)
//     5. Still writes zeros to each core's L1 at in2_cb_addr (bias/padding)
//   The DRAM kernels use InterleavedAddrGenFast<true> to compute NOC addresses
//   from a tile ID, translating it to the correct DRAM bank and offset.
//
// TRACY ZONES:
//   Each major phase is wrapped in ZoneScopedN for profiling host-side
//   bottlenecks (data gen, tilization, transfer).
//
// MODIFIED: Added DRAM branch and Tracy instrumentation (original from
// 1_compute_mm).
////////////////////////////////////////////////////////////////////////////////

// ============================================================================
// Dtype-generic pack/unpack helpers
// ============================================================================
// These encapsulate the different conversion pipelines for BFP8 and BF16:
//   BFP8:  FP32 → tilize_swizzled<float>     → pack_as_bfp8_tiles         → u32
//   BF16:  FP32 → bfloat16() → tilize_swizzled<bf16> → swizzled_to_nfaces →
//   pack_bf16 → u32
//
//   BFP8:  u32 → unpack_bfp8_tiles → untilize_swizzled<float>              →
//   FP32 BF16:  u32 → unpack_bf16 → nfaces_to_swizzled →
//   untilize_swizzled<bf16> → to_float → FP32
// ============================================================================

std::vector<uint32_t>
pack_tilized_fp32_to_device_format(const std::vector<float> &fp32_vec,
                                   uint32_t rows, uint32_t cols,
                                   const tt::DataFormat data_format) {
  if (data_format == tt::DataFormat::Bfp8_b) {
    auto tilized = tilize_swizzled(fp32_vec, rows, cols);
    return pack_as_bfp8_tiles(tt::stl::make_const_span(tilized),
                              /*row_major_input=*/true,
                              /*is_exp_a=*/false);
  } else {
    // Float16_b (BFLOAT16) path
    std::vector<bfloat16> bf16_vec;
    bf16_vec.reserve(fp32_vec.size());
    std::transform(fp32_vec.begin(), fp32_vec.end(),
                   std::back_inserter(bf16_vec),
                   [](float f) { return bfloat16(f); });
    auto tilized = tilize_swizzled(bf16_vec, rows, cols);
    auto nfaces = convert_layout_tile_swizzled_to_tile_nfaces(
        tt::stl::make_const_span(tilized));
    return pack_bfloat16_vec_into_uint32_vec(nfaces);
  }
}

std::vector<float>
unpack_device_tiles_to_fp32(const std::vector<uint32_t> &packed, uint32_t rows,
                            uint32_t cols, const tt::DataFormat data_format) {
  ZoneScopedN("Unpack Device Tiles to FP32");
  if (data_format == tt::DataFormat::Bfp8_b) {
    std::vector<float> unpacked;
    {
      ZoneScopedN("Unpack BFP8 Tiles");
      unpacked = unpack_bfp8_tiles_into_float_vec(
          packed, /*row_major_output=*/true, /*is_exp_a=*/false);
    }
    {
      ZoneScopedN("Untilize BFP8 Tiles");
      return untilize_swizzled(unpacked, rows, cols);
    }
  } else {
    // Float16_b (BFLOAT16) path
    std::vector<bfloat16> bf16_vec;
    {
      ZoneScopedN("Unpack BF16 Tiles");
      bf16_vec = unpack_uint32_vec_into_bfloat16_vec(packed);
    }
    std::vector<bfloat16> swizzled;
    {
      ZoneScopedN("BF16 Nfaces to Swizzled");
      swizzled = convert_layout_tile_nfaces_to_tile_swizzled(
          tt::stl::make_const_span(bf16_vec));
    }
    std::vector<bfloat16> untilized;
    {
      ZoneScopedN("Untilize BF16 Tiles");
      untilized = untilize_swizzled(swizzled, rows, cols);
    }
    std::vector<float> result;
    result.reserve(untilized.size());
    {
      ZoneScopedN("BF16 to FP32");
      std::transform(untilized.begin(), untilized.end(),
                     std::back_inserter(result),
                     [](const bfloat16 &v) { return static_cast<float>(v); });
    }
    return result;
  }
}

////////////////////////////////////////////////////////////////////////////////
BenchmarkInputs prepare_inputs_compute_mm(
    tt_metal::distributed::MeshDevice *device, CoreCoord core_range,
    uint32_t Mt, uint32_t Nt, uint32_t Kt, uint32_t per_core_Mt,
    uint32_t per_core_Nt, uint32_t in0_block_w, uint32_t single_tile_size,
    uint32_t in0_addr, uint32_t in1_addr, uint32_t in2_cb_addr, bool use_dram,
    uint32_t start_core_y, tt::DataFormat data_format, bool pack_device,
  bool reconstruct_effective_inputs, bool create_output_ttnn_tensor,
  bool zeros_mode) {

  ZoneScopedN("Prepare Inputs Compute MM");
  BenchmarkInputs inputs;

  // Generate input matrices. In zeros mode, fill with 0.0f to skip RNG cost
  // and make the golden trivially zero (validation becomes a max-abs scan).
  // TILE_HW = 32*32 = 1024 elements per tile.
  if (zeros_mode) {
    inputs.in0_vec.assign(static_cast<size_t>(Mt) * Kt * constants::TILE_HW,
                          0.0f);
    inputs.in1_vec.assign(static_cast<size_t>(Nt) * Kt * constants::TILE_HW,
                          0.0f);
  } else {
    // Scale factor for ML-realistic initialization: range [-5, 5]
    float scale = 5.0f;
    inputs.in0_vec = generate_fp32_random(Mt * Kt * constants::TILE_HW, scale);
    inputs.in1_vec = generate_fp32_random(Nt * Kt * constants::TILE_HW, scale);
  }

  // Zeros buffer for the "in2" circular buffer (CB index 2).
  // This is written to every core's L1 regardless of L1/DRAM mode.
  // The compute kernel reads from CB2 for bias/padding initialization;
  // we fill it with zeros since we don't use bias in this benchmark.
  std::vector<uint32_t> in2(single_tile_size / sizeof(uint32_t), 0);
  auto *target_device = device->get_devices()[0];
  (void)reconstruct_effective_inputs;

  const bool need_output_ttnn_tensor = use_dram && create_output_ttnn_tensor;
  const ttnn::DataType output_ttnn_dtype =
      (data_format == tt::DataFormat::Bfp8_b) ? ttnn::DataType::BFLOAT8_B
                                              : ttnn::DataType::BFLOAT16;

  if (pack_device && use_dram) {
    ZoneScopedN("Prepare Inputs On-Device (ttnn)");

    ttnn::Tensor host_tensor_a;
    ttnn::Tensor host_tensor_b;
    {
      ZoneScopedN("ttnn::CreateHostTensors(IN0+IN1)");
      ttnn::Shape shape_a({1, 1, Mt * 32, Kt * 32});
      host_tensor_a =
          ttnn::Tensor(tt::tt_metal::HostBuffer(inputs.in0_vec), shape_a,
                       ttnn::DataType::FLOAT32, ttnn::Layout::ROW_MAJOR);

      ttnn::Shape shape_b({1, 1, Kt * 32, Nt * 32});
      host_tensor_b =
          ttnn::Tensor(tt::tt_metal::HostBuffer(inputs.in1_vec), shape_b,
                       ttnn::DataType::FLOAT32, ttnn::Layout::ROW_MAJOR);
    }

    // ---- DRAM Step 1: Tilize and pack IN0 (full matrix A) on device ----
    {
      ZoneScopedN("ttnn::IN0_Prepare");
      ttnn::Tensor device_tensor_a_rm;
      {
        ZoneScopedN("ttnn::IN0_ToDevice");
        device_tensor_a_rm = host_tensor_a.to_device(device);
      }

      ttnn::Tensor device_tensor_a_final;
      {
        ZoneScopedN("ttnn::IN0_TilizePack");
        device_tensor_a_final =
          ttnn::tilize(device_tensor_a_rm, std::nullopt, output_ttnn_dtype);
      }

      {
        ZoneScopedN("ttnn::IN0_CaptureBufferAndRetainTensor");
        inputs.in0_buffer =
            device_tensor_a_final.device_storage().get_mesh_buffer();
        inputs.ttnn_tensors.push_back(device_tensor_a_final);
      }
    }

    // ---- DRAM Step 2: Tilize and pack IN1 (full matrix B) on device ----
    {
      ZoneScopedN("ttnn::IN1_Prepare");
      ttnn::Tensor device_tensor_b_rm;
      {
        ZoneScopedN("ttnn::IN1_ToDevice");
        device_tensor_b_rm = host_tensor_b.to_device(device);
      }

      ttnn::Tensor device_tensor_b_final;
      {
        ZoneScopedN("ttnn::IN1_TilizePack");
        device_tensor_b_final =
          ttnn::tilize(device_tensor_b_rm, std::nullopt, output_ttnn_dtype);
      }

      {
        ZoneScopedN("ttnn::IN1_CaptureBufferAndRetainTensor");
        inputs.in1_buffer =
            device_tensor_b_final.device_storage().get_mesh_buffer();
        inputs.ttnn_tensors.push_back(device_tensor_b_final);
      }
    }

    // ---- DRAM Step 3: Wait for all on-device operations to finish ----
    {
      ZoneScopedN("ttnn::Sync");
      tt::tt_metal::distributed::Finish(device->mesh_command_queue());
    }

    // ---- DRAM Step 4: Create empty output buffer via ttnn ----
    {
      ZoneScopedN("ttnn::Output_Prepare");
      ttnn::Shape shape({1, 1, Mt * 32, Nt * 32});

      ttnn::Tensor device_tensor_out;
      {
        ZoneScopedN("ttnn::Output_CreateTensor");
        device_tensor_out =
            ttnn::empty(shape, output_ttnn_dtype, ttnn::Layout::TILE, device,
                        ttnn::DRAM_MEMORY_CONFIG);
      }

      {
        ZoneScopedN("ttnn::Output_CreateTensor_Finish");
        tt::tt_metal::distributed::Finish(device->mesh_command_queue());
      }

      inputs.out_buffer = device_tensor_out.device_storage().get_mesh_buffer();
      inputs.output_ttnn_tensor = device_tensor_out;
      inputs.ttnn_tensors.push_back(device_tensor_out);
    }

    // ---- DRAM Step 5: Initialize per-core L1 zero tile for in2 ----
    {
      ZoneScopedN("ComputeMM Host CPU: MMIO L1 Zero Padding Setup");
      uint32_t num_cores_y = core_range.y;
      uint32_t num_cores_x = core_range.x;
      for (uint32_t y = 0; y < num_cores_y; y++) {
        for (uint32_t x = 0; x < num_cores_x; x++) {
          CoreCoord core = {(std::size_t)x, (std::size_t)(y + start_core_y)};
          tt_metal::detail::WriteToDeviceL1(target_device, core, in2_cb_addr,
                                            in2);
        }
      }
    }

  } else if (use_dram) {
    ZoneScopedN("Prepare DRAM Inputs");

    // ---- DRAM Step 1: Tilize and pack IN0 (full matrix A) ----
    // tilize_swizzled() converts from row-major to the hardware tile layout
    // (32x32 element groups in Z-order). pack_as_bfp8_tiles() then quantizes
    // the FP32 data into BFP8_b format (block floating point, 8-bit mantissa).
    // We tilize the ENTIRE matrix at once (Mt*32 × Kt*32), unlike L1 mode
    // which slices per-core.
    std::vector<uint32_t> in0_packed;
    uint32_t in0_num_tiles = Mt * Kt;
    uint32_t in0_size_bytes = in0_num_tiles * single_tile_size;
    {
      ZoneScopedN("ComputeMM Host CPU: Tilize and Pack IN0 (DRAM)");
      in0_packed = pack_tilized_fp32_to_device_format(inputs.in0_vec, Mt * 32,
                                                      Kt * 32, data_format);
    }

    {
      ZoneScopedN("ComputeMM Host: Create IN0 DRAM Buffer");
      // Create a DRAM interleaved buffer to hold Mt*Kt tiles.
      // InterleavedBufferConfig tells the allocator to distribute tiles
      // across DRAM banks in round-robin order. Each "page" = one tile.
      tt::tt_metal::distributed::DeviceLocalBufferConfig device_local{
          .page_size = single_tile_size,
          .buffer_type = tt_metal::BufferType::DRAM,
      };
      tt::tt_metal::distributed::ReplicatedBufferConfig global_buf{
          .size = in0_size_bytes};
      inputs.in0_buffer = tt::tt_metal::distributed::MeshBuffer::create(
          global_buf, device_local, device);
    }

    // Bandwidth-attributable H2D zone: enqueue + Finish bundled, so the
    // zone duration is the full host-visible transfer cost. BW = in0_size_bytes / zone_us.
    {
      ZoneScopedN("ComputeMM H2D Transfer IN0 (Enqueue+Finish)");
      tt::tt_metal::distributed::EnqueueWriteMeshBuffer(
          device->mesh_command_queue(), inputs.in0_buffer, in0_packed, false);
      tt::tt_metal::distributed::Finish(device->mesh_command_queue());
    }

    {
      ZoneScopedN("ComputeMM Host CPU: Decode Validation Golden IN0");
      // Update inputs.in0_vec with effective quantized values for validation
      inputs.in0_vec = unpack_device_tiles_to_fp32(in0_packed, Mt * 32, Kt * 32,
                                                   data_format);
    }

    // ---- DRAM Step 2: Tilize and pack IN1 (full matrix B) ----
    // Same flow as IN0: tilize → pack BFP8 → create DRAM buffer → write.
    // IN1 has dimensions Kt*32 (rows) × Nt*32 (cols), totaling Kt*Nt tiles.
    std::vector<uint32_t> in1_packed;
    uint32_t in1_num_tiles = Kt * Nt;
    uint32_t in1_size_bytes = in1_num_tiles * single_tile_size;
    {
      ZoneScopedN("ComputeMM Host CPU: Tilize and Pack IN1 (DRAM)");
      in1_packed = pack_tilized_fp32_to_device_format(inputs.in1_vec, Kt * 32,
                                                      Nt * 32, data_format);
    }

    {
      ZoneScopedN("ComputeMM Host: Create IN1 DRAM Buffer");
      tt::tt_metal::distributed::DeviceLocalBufferConfig device_local{
          .page_size = single_tile_size,
          .buffer_type = tt_metal::BufferType::DRAM,
      };
      tt::tt_metal::distributed::ReplicatedBufferConfig global_buf{
          .size = in1_size_bytes};
      inputs.in1_buffer = tt::tt_metal::distributed::MeshBuffer::create(
          global_buf, device_local, device);
    }

    // Bandwidth-attributable H2D zone: enqueue + Finish bundled. BW = in1_size_bytes / zone_us.
    {
      ZoneScopedN("ComputeMM H2D Transfer IN1 (Enqueue+Finish)");
      tt::tt_metal::distributed::EnqueueWriteMeshBuffer(
          device->mesh_command_queue(), inputs.in1_buffer, in1_packed, false);
      tt::tt_metal::distributed::Finish(device->mesh_command_queue());
    }

    {
      ZoneScopedN("ComputeMM Host CPU: Decode Validation Golden IN1");
      // Update inputs.in1_vec with effective quantized values for validation
      inputs.in1_vec = unpack_device_tiles_to_fp32(in1_packed, Kt * 32, Nt * 32,
                                                   data_format);
    }

    // ---- DRAM Step 3: Create empty output buffer ----
    // The output buffer holds the result C = A*B. Total tiles = Mt * Nt.
    // The kernel will write to this buffer using InterleavedAddrGenFast<true>
    // and noc_async_write_tile().
    {
      if (need_output_ttnn_tensor) {
        ZoneScopedN("ttnn::Output_CreateTensor (Unpack Device Bootstrap)");
        ttnn::Shape shape({1, 1, Mt * 32, Nt * 32});
        ttnn::Tensor device_tensor_out =
            ttnn::empty(shape, output_ttnn_dtype, ttnn::Layout::TILE, device,
                        ttnn::DRAM_MEMORY_CONFIG);

        {
          ZoneScopedN("ttnn::Output_CreateTensor_Finish");
          tt::tt_metal::distributed::Finish(device->mesh_command_queue());
        }

        inputs.out_buffer = device_tensor_out.device_storage().get_mesh_buffer();
        inputs.output_ttnn_tensor = device_tensor_out;
        inputs.ttnn_tensors.push_back(device_tensor_out);
      } else {
        ZoneScopedN("tt-metal::Output_CreateTensor DRAM Buffer (Manual)");
        uint32_t out_num_tiles = Mt * Nt;
        uint32_t out_size_bytes = out_num_tiles * single_tile_size;
        tt::tt_metal::distributed::DeviceLocalBufferConfig device_local{
            .page_size = single_tile_size,
            .buffer_type = tt_metal::BufferType::DRAM,
        };
        tt::tt_metal::distributed::ReplicatedBufferConfig global_buf{
            .size = out_size_bytes};

        inputs.out_buffer = tt::tt_metal::distributed::MeshBuffer::create(
            global_buf, device_local, device);
      }
    }

    // ---- DRAM Step 4: Initialize per-core L1 zero tile for in2 ----
    // DRAM writer kernel uses in2_cb_addr as zero source (same contract as
    // L1 writer path), so each participating core receives one zero tile.
    {
      ZoneScopedN("ComputeMM Host CPU: MMIO L1 Zero Padding Setup");
      uint32_t num_cores_y = core_range.y;
      uint32_t num_cores_x = core_range.x;
      for (uint32_t y = 0; y < num_cores_y; y++) {
        for (uint32_t x = 0; x < num_cores_x; x++) {
          CoreCoord core = {(std::size_t)x, (std::size_t)(y + start_core_y)};
          tt_metal::detail::WriteToDeviceL1(target_device, core, in2_cb_addr,
                                            in2);
        }
      }
    }

  } else {
    // ===========================================================================
    // L1 MODE (Original Logic from 1_compute_mm/test_compute_mm.cpp)
    // ===========================================================================
    // In L1 mode, each core holds its own slice of IN0 and IN1 locally in L1.
    // The host slices and tilizes per-core, then writes to each core's L1.
    // The kernels read from a fixed local L1 address using the core's own
    // NOC coordinates (phy_core.x, phy_core.y) — no DRAM access needed.
    //
    // Edge-case handling:
    //   - last_block_h: the last row of cores may have fewer tile-rows
    //   - last_block_w: the last column of cores may have fewer tile-columns
    uint32_t num_cores_y = core_range.y;
    uint32_t num_cores_x = core_range.x;

    uint32_t last_block_h =
        Mt % per_core_Mt == 0 ? per_core_Mt : Mt % per_core_Mt;
    uint32_t last_block_w =
        Nt % per_core_Nt == 0 ? per_core_Nt : Nt % per_core_Nt;

    for (int r = 0; r < (int)num_cores_y; r++) {
      // Determine actual tile-height for this core row (edge cores may be
      // smaller)
      int num_r = (r == (int)num_cores_y - 1) ? (last_block_h) : (per_core_Mt);

      // Slice and Tilize IN0 for this core row.
      // get_row_slice: extracts rows [r*per_core_Mt*32 .. r*per_core_Mt*32 +
      // num_r*32) get_col_slice: extracts the first in0_block_w*32 columns
      // (K-dim block) tilize_swizzled: converts from row-major to tile layout
      // pack_as_bfp8_tiles: quantizes FP32 → BFP8_b format
      std::vector<uint32_t> in0;
      {
        ZoneScopedN("Slicing and Tilizing IN0");
        std::vector<float> in0_slice = get_row_slice(
            inputs.in0_vec, r * per_core_Mt * 32, num_r * 32, Mt * 32, Kt * 32);
        auto in0_block_slice =
            get_col_slice(in0_slice, 0, in0_block_w * 32, num_r * 32, Kt * 32);
        in0 = pack_tilized_fp32_to_device_format(in0_block_slice, num_r * 32,
                                                 in0_block_w * 32, data_format);
      }

      for (int c = 0; c < (int)num_cores_x; c++) {
        // Determine actual tile-width for this core column
        int num_c =
            (c == (int)num_cores_x - 1) ? (last_block_w) : (per_core_Nt);

        // Generate an identity-like IN1 block for this (row, col) core.
        // In L1 mode with IN1_IS_IDENTITY define, the kernel reuses the same
        // IN1 block across all K-blocks. This block has 1.0 on the diagonal
        // so that A * I = A (passthrough for validation). It's not a real
        // full matrix multiply — use DRAM mode for realistic workloads.
        std::vector<uint32_t> in1;
        {
          ZoneScopedN("Generating and Tilizing IN1");
          std::vector<float> in1_block_slice(in0_block_w * num_c * 1024,
                                             (float)0);
          int num_ones =
              std::min(in0_block_w, static_cast<uint32_t>(num_c)) * 32;
          for (int i = 0; i < num_ones; i++) {
            in1_block_slice.at((i * (num_c * 32)) + i) = (float)1;
          }

          in1 = pack_tilized_fp32_to_device_format(
              in1_block_slice, in0_block_w * 32, num_c * 32, data_format);
        }

        // Transfer tile blocks to this core's L1 SRAM.
        // WriteToDeviceL1 writes directly to a specific address in the target
        // core's SRAM (bypassing the command queue abstraction entirely).
        {
          ZoneScopedN("ComputeMM Host CPU: MMIO L1 Data Setup");
          CoreCoord core = {(std::size_t)c, (std::size_t)(r + start_core_y)};
          tt_metal::detail::WriteToDeviceL1(target_device, core, in0_addr, in0);
          tt_metal::detail::WriteToDeviceL1(target_device, core, in1_addr, in1);
          tt_metal::detail::WriteToDeviceL1(target_device, core, in2_cb_addr,
                                            in2);
        }
      }
    }
  }

  return inputs;
}

////////////////////////////////////////////////////////////////////////////////
// create_program_compute_mm
//
// Creates the full device program for blocked matrix multiplication C = A * B.
// This function defines the entire compute graph that runs on the Tensix cores:
//
//   1. Circular Buffers (L1 scratch memory for data flow between kernels)
//   2. Data Movement Kernels (Reader for IN0/IN1, Writer for output)
//   3. Compute Kernel (FPU matrix multiply on blocks/subblocks)
//   4. Runtime Arguments (per-core addresses, strides, block sizes)
//
// The program is appended to the `program` reference; this allows
// test_sub_device_manager_mm to add multiple independent MM programs
// (one per sub-device split) into a single Program object.
//
// L1 vs DRAM KERNEL SELECTION:
//   When use_dram=false:
//     - Reader: in0_reader_bmm_tile_layout.cpp (reads from local L1)
//     - Writer: in1_reader_writer_bmm_tile_layout.cpp (reads IN1 from L1,
//               writes output to L1; IN1_IS_IDENTITY define = reuse same block)
//     - Strides are local (in0_block_w, per_core_Nt) — within the core's buffer
//
//   When use_dram=true:
//     - Reader: in0_reader_bmm_tile_layout_dram.cpp (reads from DRAM
//     interleaved)
//     - Writer: in1_reader_writer_bmm_tile_layout_dram.cpp (reads IN1 from
//     DRAM,
//               writes output to DRAM; no IN1_IS_IDENTITY — real data)
//     - Strides are GLOBAL (Kt for IN0 row stride, Nt for IN1 row stride)
//       because tiles are laid out across the full matrix in DRAM.
//     - Start tile IDs are computed per-core: core(y,x) starts at
//       in0: y * per_core_Mt * Kt    (global row offset in tile IDs)
//       in1: x * per_core_Nt          (global column offset)
//       out: y * per_core_Mt * Nt + x * per_core_Nt
//
// NOC ASSIGNMENT:
//   Reader kernel runs on RISCV_1 / NOC_0 (reads IN0 into CB0)
//   Writer kernel runs on RISCV_0 / NOC_1 (reads IN1 into CB1, writes from
//   CB16) Compute kernel runs on the FPU (Tensix math engine)
//
// 1-to-1 match with 1_compute_mm/test_compute_mm.cpp (L1 path).
// MODIFIED: Added DRAM kernel selection and DRAM runtime argument computation.
////////////////////////////////////////////////////////////////////////////////
void create_program_compute_mm(
    tt_metal::distributed::MeshDevice *device, tt::DataFormat cb_data_format,
    MathFidelity math_fidelity, bool fp32_dest_acc_en,
    uint32_t single_tile_size, CoreCoord core_range, uint32_t Mt, uint32_t Nt,
    uint32_t Kt, uint32_t in0_block_w, uint32_t out_subblock_h,
    uint32_t out_subblock_w, uint32_t per_core_Mt, uint32_t per_core_Nt,
    uint32_t out_block_h, uint32_t out_block_w, uint32_t num_blocks_h,
    uint32_t num_blocks_w, uint32_t in0_cb_addr, uint32_t in1_cb_addr,
    uint32_t in2_cb_addr, uint32_t out_cb_addr, uint32_t interm_cb_addr,
    uint32_t in0_addr, uint32_t in1_addr, uint32_t out_addr, bool use_dram,
    tt_metal::Program &program, uint32_t start_core_y) {

  // ---- Step 1: Define buffer sizes in tiles ----
  // Double buffering (num_buffer=2) allows the reader/writer to fill one
  // buffer while the compute engine processes the other, hiding data
  // movement latency behind compute.
  // CB sizes use out_block_h/w (reduced dims for DRAM multi-block)
  // In L1 mode, out_block_h == per_core_Mt and out_block_w == per_core_Nt.
  uint32_t num_buffer = 2;
  uint32_t in0_block_tiles = out_block_h * in0_block_w;
  uint32_t in0_CB_tiles = in0_block_tiles * num_buffer;
  uint32_t in1_block_tiles = out_block_w * in0_block_w;
  uint32_t in1_CB_tiles = in1_block_tiles * num_buffer;
  uint32_t out_block_tiles_count = out_block_h * out_block_w;
  uint32_t out_CB_tiles = out_block_tiles_count;
  uint32_t out_CB_size = out_CB_tiles * single_tile_size;

  // ---- Step 2: Compute kernel compile-time arguments ----
  // These arguments configure the FPU (Tensix math engine) for block matrix
  // multiplication. The FPU processes data in "subblocks" that fit into
  // its DST (destination) registers.
  //
  // num_blocks:             how many K-dimension blocks to iterate over
  // in0_num_subblocks:      per_core_Mt / out_subblock_h
  // in0_block_num_tiles:    total tiles in one IN0 block
  // in0_subblock_num_tiles: tiles in one subblock of IN0
  // out_subblock_h/w:       subblock dimensions for register accumulation
  // out_subblock_num_tiles: tiles in one output subblock (must fit in DST)
  // Subblocks are computed from out_block_h/w (NOT per_core_Mt/Nt)
  uint32_t num_blocks = (Kt / in0_block_w);
  uint32_t in0_num_subblocks = (out_block_h / out_subblock_h);
  uint32_t in0_block_num_tiles =
      out_subblock_h * in0_block_w * in0_num_subblocks;
  uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;
  uint32_t in1_num_subblocks = (out_block_w / out_subblock_w);
  uint32_t in1_block_num_tiles =
      out_subblock_w * in0_block_w * in1_num_subblocks;
  uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;
  uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

  // Compute kernel args: 14 args (0-13)
  // Args 12-13 are the TTNN-aligned outer block loop dimensions.
  // When num_blocks_h/w == 1, the kernel behaves identically to before.
  std::vector<uint32_t> compute_kernel_args = {
      in0_block_w,            // 0
      in0_num_subblocks,      // 1
      in0_block_num_tiles,    // 2
      in0_subblock_num_tiles, // 3
      in1_num_subblocks,      // 4
      in1_block_num_tiles,    // 5
      in1_per_core_w,         // 6
      num_blocks,             // 7
      out_subblock_h,         // 8
      out_subblock_w,         // 9
      out_subblock_num_tiles, // 10
      1,                      // 11: batch
      num_blocks_w,           // 12: out_num_blocks_x
      num_blocks_h};          // 13: out_num_blocks_y

  CoreRange all_cores({(std::size_t)0, (std::size_t)start_core_y},
                      {(std::size_t)core_range.x - 1,
                       (std::size_t)(core_range.y - 1 + start_core_y)});

  // ---- Step 3: Create Circular Buffers (CBs) ----
  // Circular Buffers are L1 memory regions that pipeline data between the
  // reader/writer kernels and the compute kernel.
  //   CB0 (c_0):  IN0 block (Matrix A)   — double-buffered
  //   CB1 (c_1):  IN1 block (Matrix B)   — double-buffered
  //   CB2 (c_2):  Bias/padding tiles     — 2 tiles (shared zeros)
  //   CB16 (c_16): Output accumulation   — compute writes here
  //   CB24 (c_24): Output staging        — same memory as CB16 (aliased)
  //
  // The compute kernel reads from CB0/CB1, accumulates in DST registers,
  // and writes results to CB16. The writer kernel reads from CB16 and
  // writes to L1 or DRAM.
  tt_metal::CircularBufferConfig cb_src0 =
      tt_metal::CircularBufferConfig(in0_CB_tiles * single_tile_size,
                                     {{tt::CBIndex::c_0, cb_data_format}})
          .set_page_size(tt::CBIndex::c_0, single_tile_size);
  tt_metal::CreateCircularBuffer(program, all_cores, cb_src0);

  tt_metal::CircularBufferConfig cb_src1 =
      tt_metal::CircularBufferConfig(in1_CB_tiles * single_tile_size,
                                     {{tt::CBIndex::c_1, cb_data_format}})
          .set_page_size(tt::CBIndex::c_1, single_tile_size);
  tt_metal::CreateCircularBuffer(program, all_cores, cb_src1);

  tt_metal::CircularBufferConfig cb_src2 =
      tt_metal::CircularBufferConfig(2 * single_tile_size,
                                     {{tt::CBIndex::c_2, cb_data_format}})
          .set_page_size(tt::CBIndex::c_2, single_tile_size);
  tt_metal::CreateCircularBuffer(program, all_cores, cb_src2);

  // Output + intermediate CBs are configured together to match upstream
  // 1_compute_mm behavior for bmm_large_block_zm_fused_bias_activation.
  std::map<uint8_t, tt::DataFormat> cb_out_config_map = {
      {(uint8_t)tt::CBIndex::c_16, cb_data_format},
      {(uint8_t)tt::CBIndex::c_24, cb_data_format}};
  tt_metal::CircularBufferConfig cb_out_config =
      tt_metal::CircularBufferConfig(out_CB_size, cb_out_config_map)
          .set_page_size(tt::CBIndex::c_16, single_tile_size)
          .set_page_size(tt::CBIndex::c_24, single_tile_size);
  tt_metal::CreateCircularBuffer(program, CoreRangeSet({all_cores}),
                                 cb_out_config);

  // ---- Step 4: Create Data Movement and Compute Kernels ----
  // The kernel binary path determines whether data is read from L1 or DRAM.
  //
  // DRAM kernels use InterleavedAddrGenFast<true> in the kernel code to
  // translate tile IDs into DRAM NOC addresses. They read the full matrix
  // from a global interleaved buffer shared across all cores.
  //
  // L1 kernels read from a fixed local L1 address. The IN1_IS_IDENTITY
  // define tells the L1 writer kernel to reuse the same IN1 block for
  // every K-block iteration (since in L1 mode we load an identity matrix).
  std::string reader_kernel_path;
  std::string writer_kernel_path;
  std::map<std::string, std::string> writer_defines;

  if (use_dram) {
    // DRAM kernels: read real data from DRAM interleaved buffers.
    // These kernels use InterleavedAddrGenFast<true> for address generation.
    // No IN1_IS_IDENTITY — every K-block reads fresh data from DRAM.
    // NOTE: The shell script (run_full_charac.sh) copies these from
    //       kernels_common/ into kernels/ when --dram is specified.
    reader_kernel_path =
      resolve_kernel_source_path("in0_reader_bmm_tile_layout_dram.cpp");
    writer_kernel_path = resolve_kernel_source_path(
      "in1_reader_writer_bmm_tile_layout_dram.cpp");
    // No IN1_IS_IDENTITY for DRAM — we read real matrix data from DRAM
  } else {
    // L1 kernels: read from local L1 SRAM.
    // IN1_IS_IDENTITY define tells the writer kernel to reuse the same
    // IN1 block for all K-block iterations (identity matrix workaround).
    // NOTE: The shell script copies these from kernels_common/ into kernels/.
    reader_kernel_path = resolve_kernel_source_path("in0_reader_bmm_tile_layout.cpp");
    writer_kernel_path =
      resolve_kernel_source_path("in1_reader_writer_bmm_tile_layout.cpp");
    writer_defines["IN1_IS_IDENTITY"] = "1";
  }

  auto mm_reader_id = tt_metal::CreateKernel(
      program, reader_kernel_path, all_cores,
      tt_metal::DataMovementConfig{.processor =
                                       tt_metal::DataMovementProcessor::RISCV_1,
                                   .noc = tt_metal::NOC::RISCV_0_default});

  auto mm_writer_id = tt_metal::CreateKernel(
      program, writer_kernel_path, all_cores,
      tt_metal::DataMovementConfig{.processor =
                                       tt_metal::DataMovementProcessor::RISCV_0,
                                   .noc = tt_metal::NOC::RISCV_1_default,
                                   .defines = writer_defines});

  // Compute kernel: Tensix FPU math engine (same for both L1/DRAM modes).
  // The compute kernel operates entirely on L1 circular buffers,
  // so it doesn't need to know whether data came from L1 or DRAM.
  // NOTE: Shell script always copies this from kernels_common/ into kernels/.
  tt_metal::CreateKernel(
      program,
      resolve_kernel_source_path("bmm_large_block_zm_fused_bias_activation.cpp"),
      all_cores,
      tt_metal::ComputeConfig{.math_fidelity = math_fidelity,
                              .fp32_dest_acc_en = fp32_dest_acc_en,
                              .compile_args = compute_kernel_args});

  // ---- Step 5: Set Runtime Arguments per core ----
  // Each core needs to know which portion of the matrix to process.
  // The key difference between L1 and DRAM modes:
  //
  // L1 MODE:
  //   - in0_addr/in1_addr point to the core's own L1 memory
  //   - start_tile_id = 0 (data starts at beginning of local buffer)
  //   - stride_h = in0_block_w (stride within the local block)
  //   - All cores use the same address; data was pre-loaded per-core
  //
  // DRAM MODE:
  //   - in0_addr/in1_addr/out_addr point to DRAM buffer base addresses
  //   - start_tile_id is computed per-core using GLOBAL matrix coordinates:
  //       IN0: y * per_core_Mt * Kt  (row offset into full A matrix)
  //       IN1: x * per_core_Nt       (column offset into full B matrix)
  //       OUT: y * per_core_Mt * Nt + x * per_core_Nt
  //   - stride_h uses GLOBAL width (Kt for IN0, Nt for IN1/OUT)
  //   - next_block_stride for IN0 = in0_block_w (advance along K in tiles)
  //   - next_block_stride for IN1 = in0_block_w * Nt (skip Nt columns per
  //   K-row)
  //
  // last_block_h/w handle edge cores that may have fewer tiles.
  uint32_t last_block_h =
      Mt % per_core_Mt == 0 ? per_core_Mt : Mt % per_core_Mt;
  uint32_t last_block_w =
      Nt % per_core_Nt == 0 ? per_core_Nt : Nt % per_core_Nt;

  for (int y = 0; y < (int)core_range.y; y++) {
    for (int x = 0; x < (int)core_range.x; x++) {
      CoreCoord core = {(std::size_t)x, (std::size_t)(y + start_core_y)};
      auto phy_core = device->worker_core_from_logical_core(core);

      uint32_t cur_core_valid_Mt =
          (y == (int)core_range.y - 1) ? last_block_h : per_core_Mt;
      uint32_t cur_core_valid_Nt =
          (x == (int)core_range.x - 1) ? last_block_w : per_core_Nt;

      // Per-core boundary metadata (aligned with upstream 1_compute_mm)
      // For DRAM multi-block mode, these describe ONLY the last outer block
      // in each dimension. Interior outer blocks remain full-sized.
      uint32_t last_outer_block_h = (cur_core_valid_Mt % out_block_h == 0)
                                        ? out_block_h
                                        : (cur_core_valid_Mt % out_block_h);
      uint32_t last_outer_block_w = (cur_core_valid_Nt % out_block_w == 0)
                                        ? out_block_w
                                        : (cur_core_valid_Nt % out_block_w);

      uint32_t out_num_subblocks_h = out_block_h / out_subblock_h;
      uint32_t out_num_subblocks_w = out_block_w / out_subblock_w;

      uint32_t last_outer_num_nonzero_subblocks_h =
          ((last_outer_block_h - 1) / out_subblock_h) + 1;
      uint32_t last_outer_num_nonzero_subblocks_w =
          ((last_outer_block_w - 1) / out_subblock_w) + 1;

      uint32_t last_outer_subblock_h =
          (last_outer_block_h % out_subblock_h == 0)
              ? out_subblock_h
              : (last_outer_block_h % out_subblock_h);
      uint32_t last_outer_subblock_w =
          (last_outer_block_w % out_subblock_w == 0)
              ? out_subblock_w
              : (last_outer_block_w % out_subblock_w);

      uint32_t last_outer_padded_subblock_tiles_addr_skip =
          single_tile_size * (out_subblock_w - last_outer_subblock_w);
      uint32_t last_outer_padded_block_tiles_w_skip =
          out_subblock_num_tiles *
          (out_num_subblocks_w - last_outer_num_nonzero_subblocks_w);
      uint32_t last_outer_padded_block_tiles_h_skip =
          (out_num_subblocks_h - last_outer_num_nonzero_subblocks_h) *
          (out_block_w * out_subblock_h);

      std::vector<uint32_t> reader_args;
      std::vector<uint32_t> writer_args;

      if (use_dram) {
        // === DRAM Reader Args (16 args) ===
        // Matches in0_reader_bmm_tile_layout_dram.cpp get_arg_val indices
        uint32_t in0_start_tile_id = y * per_core_Mt * Kt; // global row * Kt
        reader_args = {
            in0_addr,                  // 0: in0_tensor_addr
            in0_start_tile_id,         // 1: in0_tensor_start_tile_id
            1,                         // 2: in0_tensor_stride_w (tiles)
            Kt,                        // 3: in0_tensor_stride_h (full K width)
            in0_block_w,               // 4: in0_tensor_next_block_stride
            in0_block_w,               // 5: in0_block_w
            out_block_h,               // 6: in0_block_h (uses out_block_h)
            in0_block_w * out_block_h, // 7: in0_block_num_tiles
            num_blocks,                // 8: num_blocks
            (uint32_t)phy_core.x,      // 9: noc_x
            (uint32_t)phy_core.y,      // 10: noc_y
            last_outer_block_h,        // 11: last_block_h for boundary bh
            num_blocks_h,              // 12: num_blocks_h_dim
            num_blocks_w,              // 13: num_blocks_w_dim
            out_block_h * Kt,          // 14: in0_h_dim_stride
            in2_cb_addr,               // 15: in2_cb_addr (zeros source)
        };

        // === DRAM Writer Args (36 args) ===
        // Matches in1_reader_writer_bmm_tile_layout_dram.cpp get_arg_val
        // indices
        uint32_t in1_start_tile_id = x * per_core_Nt; // global col
        uint32_t out_start_tile_id = y * per_core_Mt * Nt + x * per_core_Nt;

        writer_args = {
            in1_addr,                  // 0: in1_tensor_addr
            in1_start_tile_id,         // 1: in1_tensor_start_tile_id
            1,                         // 2: in1_tensor_stride_w
            Nt,                        // 3: in1_tensor_stride_h (full N width)
            in0_block_w * Nt,          // 4: in1_tensor_next_block_stride
            out_block_w,               // 5: in1_block_w (uses out_block_w)
            in0_block_w,               // 6: in1_block_h
            out_block_w * in0_block_w, // 7: in1_block_num_tiles
            num_blocks,                // 8: num_blocks
            in2_cb_addr,               // 9: in2_cb_addr
            (uint32_t)phy_core.x,      // 10: noc_x
            (uint32_t)phy_core.y,      // 11: noc_y
            out_addr,                  // 12: out_tensor_addr
            out_start_tile_id,         // 13: out_tensor_start_tile_id
            1,                         // 14: out_tensor_stride_w
            Nt,                        // 15: out_tensor_stride_h (full N width)
            out_subblock_w,            // 16: out_tensor_next_subblock_stride_w
            out_subblock_h * Nt,       // 17: out_tensor_next_subblock_stride_h
            out_subblock_w,            // 18: out_subblock_w
            out_subblock_h,            // 19: out_subblock_h
            out_subblock_num_tiles,    // 20: out_subblock_tile_count
            in1_num_subblocks,         // 21: out_num_subblocks_w
            in0_num_subblocks,         // 22: out_num_subblocks_h
            last_outer_block_w,        // 23: last_block_w for boundary bw
            last_outer_num_nonzero_subblocks_h,   // 24: last bh nonzero sbh
            last_outer_subblock_h,                // 25: last bh tail sbh height
            last_outer_padded_block_tiles_h_skip, // 26: last bh padded rows
            last_outer_num_nonzero_subblocks_w,   // 27: last bw nonzero sbw
            last_outer_subblock_w,                // 28: last bw tail sbw width
            last_outer_padded_subblock_tiles_addr_skip, // 29: tail sbw skip
            last_outer_padded_block_tiles_w_skip,       // 30: last bw pad skip
            num_blocks_h,                               // 31: num_blocks_h_dim
            num_blocks_w,                               // 32: num_blocks_w_dim
            out_block_w,                                // 33: in1_w_dim_stride
            out_block_h * Nt,                           // 34: out_h_dim_stride
            out_block_w,                                // 35: out_w_dim_stride
        };

        if (x == 0 && y == 0) {
          uint32_t in0_total_tiles = Mt * Kt;
          uint32_t in1_total_tiles = Kt * Nt;
          uint32_t out_total_tiles = Mt * Nt;

          uint32_t in0_last_tile_est =
              in0_start_tile_id + ((num_blocks_h - 1) * out_block_h * Kt) +
              ((num_blocks - 1) * in0_block_w) + ((out_block_h - 1) * Kt) +
              (in0_block_w - 1);

          uint32_t in1_last_tile_est =
              in1_start_tile_id + ((num_blocks_w - 1) * out_block_w) +
              ((num_blocks - 1) * (in0_block_w * Nt)) +
              ((in0_block_w - 1) * Nt) + (out_block_w - 1);

          uint32_t out_last_tile_est =
              out_start_tile_id + ((num_blocks_h - 1) * (out_block_h * Nt)) +
              ((num_blocks_w - 1) * out_block_w) + ((out_block_h - 1) * Nt) +
              (out_block_w - 1);

          log_info(LogTest,
                   "DRAM core(0,0) ranges: in0_start={}, in0_last_est={}, "
                   "in1_start={}, in1_last_est={}, out_start={}, "
                   "out_last_est={}",
                   in0_start_tile_id, in0_last_tile_est, in1_start_tile_id,
                   in1_last_tile_est, out_start_tile_id, out_last_tile_est);

          log_info(LogTest,
                   "DRAM core(0,0) totals: in0_total={}, in1_total={}, "
                   "out_total={}, num_blocks={}x{} outer, K_blocks={}",
                   in0_total_tiles, in1_total_tiles, out_total_tiles,
                   num_blocks_h, num_blocks_w, num_blocks);
        }

      } else {
        // === L1 Reader Args (12 args — upstream 1_compute_mm layout) ===
        reader_args = {in0_addr,
                       0,           // start_tile_id (local buffer)
                       1,           // stride_w
                       in0_block_w, // stride_h
                       in0_block_w, // next_block_stride
                       in0_block_w, // block_w
                       per_core_Mt, // block_h
                       in0_block_w * per_core_Mt, // block_num_tiles
                       num_blocks,
                       (uint32_t)phy_core.x,
                       (uint32_t)phy_core.y,
                       cur_core_valid_Mt};

        // === L1 Writer Args (31 args — upstream 1_compute_mm layout) ===
        writer_args = {
            in1_addr,
            0,
            1,
            cur_core_valid_Nt,
            in0_block_w * per_core_Nt,
            per_core_Nt,
            in0_block_w,
            per_core_Nt * in0_block_w,
            num_blocks,
            in2_cb_addr,
            (uint32_t)phy_core.x,
            (uint32_t)phy_core.y,
            out_addr,
            0,
            1,
            cur_core_valid_Nt,
            out_subblock_w,
            out_subblock_h * cur_core_valid_Nt,
            out_subblock_w,
            out_subblock_h,
            out_subblock_w * out_subblock_h,
            per_core_Nt / out_subblock_w,
            per_core_Mt / out_subblock_h,
            cur_core_valid_Nt,
            (y == (int)core_range.y - 1) ? last_outer_num_nonzero_subblocks_h
                                         : (per_core_Mt / out_subblock_h),
            (y == (int)core_range.y - 1) ? last_outer_subblock_h
                                         : out_subblock_h,
            (y == (int)core_range.y - 1) ? last_outer_padded_block_tiles_h_skip
                                         : 0,
            (x == (int)core_range.x - 1) ? last_outer_num_nonzero_subblocks_w
                                         : (per_core_Nt / out_subblock_w),
            (x == (int)core_range.x - 1) ? last_outer_subblock_w
                                         : out_subblock_w,
            (x == (int)core_range.x - 1)
                ? last_outer_padded_subblock_tiles_addr_skip
                : 0,
            (x == (int)core_range.x - 1) ? last_outer_padded_block_tiles_w_skip
                                         : 0};
      }

      tt_metal::SetRuntimeArgs(program, mm_reader_id, core, reader_args);
      tt_metal::SetRuntimeArgs(program, mm_writer_id, core, writer_args);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////
/// Compute MM Test
///////////////////////////////////////////////////////////////////////////////////////////
bool test_compute_mm(tt::tt_metal::distributed::MeshDevice *device,
                     const TestParams &params) {
  bool pass = true;
  try {
    ZoneScopedN("ComputeMM Functional Blocks");
    log_info(LogTest, "Starting Compute MM Test");
    log_info(LogTest, "M={}, N={}, K={}", params.M, params.N, params.K);
    log_info(LogTest, "Dispatch mode: direct enqueue per iteration");

    auto arch = device->arch();
    uint32_t l1_size = 0;
    uint32_t l1_unreserved_base = 0;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    bool fp32_dest_acc_en = false;
    uint32_t Mt = 0, Nt = 0, Kt = 0;
    uint32_t num_cores_x = params.core_x;
    uint32_t num_cores_y = params.core_y;
    CoreCoord core_range(num_cores_x, num_cores_y);
    uint32_t per_core_Mt = 0;
    uint32_t per_core_Nt = 0;
    tt::DataFormat data_format = tt::DataFormat::Bfp8_b;
    uint32_t single_tile_size = 0;
    uint32_t out_block_h = 0, out_block_w = 0, in0_block_w = 0;
    uint32_t num_blocks_h = 1, num_blocks_w = 1;
    uint32_t out_subblock_h = 0, out_subblock_w = 0;
    uint32_t in0_cb_addr = 0, in1_cb_addr = 0, in2_cb_addr = 0, out_cb_addr = 0,
             interm_cb_addr = 0, in0_addr = 0, in1_addr = 0, out_addr = 0;
    BenchmarkInputs inputs;
    uint32_t effective_in0_addr = 0;
    uint32_t effective_in1_addr = 0;
    uint32_t effective_out_addr = 0;

    {
      ZoneScopedN("ComputeMM Input Data Processing");

      {
        ZoneScopedN("ComputeMM Host Setup and Blocking");
        l1_size = get_l1_size(arch);
        l1_unreserved_base =
            device->allocator()->get_base_allocator_addr(HalMemType::L1);
        std::tie(math_fidelity, fp32_dest_acc_en) = get_compute_params(arch);

        std::tie(Mt, Nt, Kt) =
            get_aligned_input_tile_num(params.M, params.N, params.K);

        per_core_Mt = ((Mt - 1) / num_cores_y) + 1;
        per_core_Nt = ((Nt - 1) / num_cores_x) + 1;

        data_format = compute_data_format_from_input_dtype(params.input_dtype);
        single_tile_size = tt::tile_size(data_format);

        if (params.use_dram) {
          auto [obh, obw, bw] = get_dynamic_l1_block_params(
              per_core_Mt, per_core_Nt, Kt, single_tile_size, l1_size,
              l1_unreserved_base);
          out_block_h = obh;
          out_block_w = obw;
          in0_block_w = bw;
          num_blocks_h = per_core_Mt / out_block_h;
          num_blocks_w = per_core_Nt / out_block_w;

          log_info(LogTest,
                   "DRAM blocking: per_core={}x{}, out_block={}x{}, "
                   "num_blocks={}x{}, in0_block_w={}",
                   per_core_Mt, per_core_Nt, out_block_h, out_block_w,
                   num_blocks_h, num_blocks_w, in0_block_w);
        } else {
          out_block_h = per_core_Mt;
          out_block_w = per_core_Nt;
          in0_block_w =
              get_in0_block_w(per_core_Mt, per_core_Nt, Kt, single_tile_size,
                              l1_size, l1_unreserved_base, false);
        }

        if (in0_block_w == 0) {
          uint32_t out_cb_tiles = out_block_h * out_block_w;
          uint32_t out_cb_bytes = out_cb_tiles * single_tile_size;
          uint32_t avail_l1 = l1_size - l1_unreserved_base;
          log_error(LogTest,
                    "Insufficient L1 memory for M={}, N={}, K={} "
                    "(out_block={}x{}, out_CB={}tiles={}KB, "
                    "avail_L1={}KB, dram={})",
                    params.M, params.N, params.K, out_block_h, out_block_w,
                    out_cb_tiles, out_cb_bytes / 1024, avail_l1 / 1024,
                    params.use_dram ? "yes" : "no");
          return false;
        }

        std::tie(out_subblock_h, out_subblock_w) =
            get_out_subblock_params(out_block_h, out_block_w);

        std::tie(in0_cb_addr, in1_cb_addr, in2_cb_addr, out_cb_addr,
                 interm_cb_addr, in0_addr, in1_addr, out_addr) =
            get_all_buffers_addresses(out_block_h, out_block_w, in0_block_w,
                                      single_tile_size, l1_unreserved_base,
                                      params.use_dram);
      }

      {
        ZoneScopedN("ComputeMM Host Prepare Inputs");
        inputs = prepare_inputs_compute_mm(
            device, core_range, Mt, Nt, Kt, per_core_Mt, per_core_Nt,
            in0_block_w, single_tile_size, in0_addr, in1_addr, in2_cb_addr,
            params.use_dram, 0, data_format, params.pack_device,
          !params.bypass_check,
          params.unpack_device,
          params.zeros_mode);
      }

      {
        ZoneScopedN("ComputeMM Host Resolve Buffer Addresses");
        effective_in0_addr = in0_addr;
        effective_in1_addr = in1_addr;
        effective_out_addr = out_addr;
        if (params.use_dram) {
          // MeshBuffer doesn't expose address() directly; extract the
          // underlying per-device Buffer* via get_device_buffer().
          auto coord = tt::tt_metal::distributed::MeshCoordinate(0, 0);
          effective_in0_addr =
              inputs.in0_buffer->get_device_buffer(coord)->address();
          effective_in1_addr =
              inputs.in1_buffer->get_device_buffer(coord)->address();
          effective_out_addr =
              inputs.out_buffer->get_device_buffer(coord)->address();
          log_info(LogTest, "DRAM mode: in0=0x{:x}, in1=0x{:x}, out=0x{:x}",
                   effective_in0_addr, effective_in1_addr, effective_out_addr);
        }
      }
    }

    // 6. Create Program and Run (single program — kernel handles multi-block)
    log_info(LogTest, "Num tests {}", params.num_iters);

    tt_metal::Program program;
    {
      ZoneScopedN("ComputeMM Host Program Build");
      create_program_compute_mm(
          device, data_format, math_fidelity, fp32_dest_acc_en,
          single_tile_size, core_range, Mt, Nt, Kt, in0_block_w, out_subblock_h,
          out_subblock_w, per_core_Mt, per_core_Nt, out_block_h, out_block_w,
          num_blocks_h, num_blocks_w, in0_cb_addr, in1_cb_addr, in2_cb_addr,
          out_cb_addr, interm_cb_addr, effective_in0_addr, effective_in1_addr,
          effective_out_addr, params.use_dram, program);
    }

    auto mesh_workload = tt_metal::distributed::MeshWorkload();
    mesh_workload.add_program(
        tt::tt_metal::distributed::MeshCoordinateRange{{0, 0}, {0, 0}},
        std::move(program));

    {
      ZoneScopedN("ComputeMM Host Dispatch");
      for (uint32_t i = 0; i < params.num_iters; ++i) {
        ZoneScopedN("ComputeMM Host Dispatch Iteration");
        ZoneValue(i);
        {
          ZoneScopedN("ComputeMM Host Enqueue");
          tt_metal::distributed::EnqueueMeshWorkload(
              device->mesh_command_queue(), mesh_workload, false);
        }
        {
          ZoneScopedN("ComputeMM Host FinishWait");
          tt_metal::distributed::Finish(device->mesh_command_queue());
        }
      }
    }

    if (!params.bypass_check) {
      ZoneScopedN("ComputeMM Host Post Processing");
      log_info(LogTest, "Validation Started...");
      log_validation_pipeline_steps(params);

      if (params.use_dram && params.pack_device && !params.zeros_mode) {
        ZoneScopedN("ComputeMM Host Validation: Reconstruct Effective Inputs");

        // Reconstruct effective packed inputs for rigorous golden-reference
        // validation. This aligns device-pack path with cpu-pack semantics.
        std::vector<uint32_t> in0_packed_effective;
        {
          ZoneScopedN("ComputeMM Host Validation: Read Packed IN0 (Device Pack)");
          tt::tt_metal::distributed::ReadShard(
              device->mesh_command_queue(), in0_packed_effective,
              inputs.in0_buffer, tt::tt_metal::distributed::MeshCoordinate(0, 0),
              true);
        }
        {
          ZoneScopedN("ComputeMM Host Validation: Decode Effective IN0 (Device Pack)");
          inputs.in0_vec = unpack_device_tiles_to_fp32(in0_packed_effective,
                                                       Mt * 32, Kt * 32,
                                                       data_format);
        }

        std::vector<uint32_t> in1_packed_effective;
        {
          ZoneScopedN("ComputeMM Host Validation: Read Packed IN1 (Device Pack)");
          tt::tt_metal::distributed::ReadShard(
              device->mesh_command_queue(), in1_packed_effective,
              inputs.in1_buffer, tt::tt_metal::distributed::MeshCoordinate(0, 0),
              true);
        }
        {
          ZoneScopedN("ComputeMM Host Validation: Decode Effective IN1 (Device Pack)");
          inputs.in1_vec = unpack_device_tiles_to_fp32(in1_packed_effective,
                                                       Kt * 32, Nt * 32,
                                                       data_format);
        }
      }

      // Compute Golden Reference (or trivially all zeros in zeros_mode).
      std::vector<float> golden_vec;
      {
        ZoneScopedN("ComputeMM Host Golden Reference");
        if (params.zeros_mode) {
          log_info(LogTest, "Golden Reference: all-zeros (zeros mode)");
          golden_vec.assign(static_cast<size_t>(params.M) * params.N, 0.0f);
        } else {
          log_info(LogTest, "Computing Golden Reference (FP32)...");
          golden_vec = matmul_reference(inputs.in0_vec, inputs.in1_vec,
                                        params.M, params.N, params.K);
        }
      }

      // Read Back Results
      log_info(LogTest, "Reading Device Results...");
      std::vector<float> device_vec(params.M * params.N, 0.0f);
        const bool output_needs_trim =
          ((params.M % constants::TILE_HEIGHT) != 0) ||
          ((params.N % constants::TILE_WIDTH) != 0);
      auto *target_device = device->get_devices()[0];

      {
        ZoneScopedN("ComputeMM Host Device Readback");
        if (params.use_dram) {
          if (params.unpack_device) {
            ZoneScopedN("ComputeMM Host Device Unpack (ttnn)");
            if (!inputs.output_ttnn_tensor.has_value()) {
              throw std::runtime_error(
                  "Device unpack requested, but no output TTNN tensor was "
                  "created. Ensure output tensor bootstrap is enabled.");
            }
            auto &device_tensor_out = inputs.output_ttnn_tensor.value();

            if (params.output_dtype_mode == OutputDTypeMode::Fp32) {
              // 1. Typecast to FLOAT32 while still in TILE layout
              ttnn::Tensor device_tensor_fp32_tile;
              {
                ZoneScopedN("ttnn::typecast(FLOAT32)");
                device_tensor_fp32_tile =
                    ttnn::typecast(device_tensor_out, ttnn::DataType::FLOAT32);
              }

              // 2. Convert to ROW_MAJOR layout on device
              ttnn::Tensor device_tensor_fp32_rm;
              {
                ZoneScopedN("ttnn::untilize");
                device_tensor_fp32_rm = ttnn::untilize(device_tensor_fp32_tile);
              }

              {
                ZoneScopedN("ttnn::untilize_Finish");
                tt::tt_metal::distributed::Finish(device->mesh_command_queue());
              }

              // 3. Bring to host as vector<float>
              {
                ZoneScopedN("ttnn::to_vector<float> (Readback)");
                device_vec = device_tensor_fp32_rm.to_vector<float>();
              }
            } else {
              // Native/BF16 export path: untilize directly, no explicit FP32 cast.
              ttnn::Tensor device_tensor_rm;
              {
                ZoneScopedN("ttnn::untilize");
                device_tensor_rm = ttnn::untilize(device_tensor_out);
              }
              {
                ZoneScopedN("ttnn::untilize_Finish");
                tt::tt_metal::distributed::Finish(device->mesh_command_queue());
              }
              {
                ZoneScopedN("ttnn::to_vector<float> (Readback)");
                device_vec = device_tensor_rm.to_vector<float>();
              }
            }

            // Trim to actual M x N if needed
            if (output_needs_trim &&
                device_vec.size() > (size_t)(params.M * params.N)) {
              {
                ZoneScopedN("ComputeMM Host Trim Device Readback to MxN");
                std::vector<float> trimmed(params.M * params.N, 0.0f);
                for (uint32_t r = 0; r < params.M; ++r) {
                  for (uint32_t c = 0; c < params.N; ++c) {
                    trimmed[r * params.N + c] = device_vec[r * (Nt * 32) + c];
                  }
                }
                device_vec = trimmed;
              }
            }
          } else {
            // === DRAM Readback ===
            // Read entire output buffer from DRAM
            std::vector<uint32_t> out_data;
            {
              ZoneScopedN("ComputeMM Host ReadShard DRAM Output");
              tt::tt_metal::distributed::ReadShard(
                  device->mesh_command_queue(), out_data, inputs.out_buffer,
                  tt::tt_metal::distributed::MeshCoordinate(0, 0), true);
            }

            {
              ZoneScopedN("ComputeMM Host Decode DRAM");
              // Unpack and untilize the full output
              device_vec = unpack_device_tiles_to_fp32(out_data, Mt * 32,
                                                       Nt * 32, data_format);
              // Trim to actual M x N if needed
              if (output_needs_trim &&
                  device_vec.size() > (size_t)(params.M * params.N)) {
                {
                  ZoneScopedN("ComputeMM Host Trim DRAM Decode to MxN");
                  std::vector<float> trimmed(params.M * params.N, 0.0f);
                  for (uint32_t r = 0; r < params.M; ++r) {
                    for (uint32_t c = 0; c < params.N; ++c) {
                      trimmed[r * params.N + c] = device_vec[r * (Nt * 32) + c];
                    }
                  }
                  device_vec = trimmed;
                }
              }
            }
          }
        } else {
          // === L1 Readback (per-core: read all raw tiles first) ===
          std::vector<std::vector<uint32_t>> all_core_tiles(num_cores_y *
                                                            num_cores_x);
          {
            ZoneScopedN("ComputeMM Host ReadFromDeviceL1 All Cores");
            for (int y = 0; y < (int)num_cores_y; y++) {
              for (int x = 0; x < (int)num_cores_x; x++) {
                CoreCoord core = {(std::size_t)x, (std::size_t)y};
                uint32_t core_n_tiles = per_core_Mt * per_core_Nt;
                uint32_t read_size = core_n_tiles * single_tile_size;

                tt_metal::detail::ReadFromDeviceL1(
                    target_device, core, out_addr, read_size,
                    all_core_tiles[y * num_cores_x + x]);
              }
            }
          }

          {
            ZoneScopedN("ComputeMM Host Decode L1");
            // Unpack & stitch all cores
            for (int y = 0; y < (int)num_cores_y; y++) {
              for (int x = 0; x < (int)num_cores_x; x++) {
                auto core_data_untilized = unpack_device_tiles_to_fp32(
                    all_core_tiles[y * num_cores_x + x], per_core_Mt * 32,
                    per_core_Nt * 32, data_format);

                uint32_t global_r_start = y * per_core_Mt * 32;
                uint32_t global_c_start = x * per_core_Nt * 32;
                uint32_t r_len = per_core_Mt * 32;
                uint32_t c_len = per_core_Nt * 32;

                for (uint32_t r = 0; r < r_len; ++r) {
                  for (uint32_t c = 0; c < c_len; ++c) {
                    uint32_t global_idx = (global_r_start + r) * (params.N) +
                                          (global_c_start + c);
                    if (global_idx < device_vec.size()) {
                      device_vec[global_idx] =
                          core_data_untilized[r * c_len + c];
                    }
                  }
                }
              }
            }
          }
        }
      }

      if (params.output_dtype_mode == OutputDTypeMode::Bf16) {
        ZoneScopedN("ComputeMM Host Output DType Postprocess BF16");
        for (auto &v : device_vec) {
          v = static_cast<float>(bfloat16(v));
        }
      }

      if (params.zeros_mode) {
        float max_abs = 0.0f;
        {
          ZoneScopedN("ComputeMM Host Validation Metrics");
          max_abs = get_max_abs(device_vec);
        }
        log_validation_sample_pairs("ComputeMM Validation (zeros)", golden_vec,
                                    device_vec, 12);
        const float zeros_eps = 1e-6f;
        log_info(LogTest,
                 "Validation Result (zeros mode): max_abs={:.6e}, eps={:.1e}",
                 max_abs, zeros_eps);
        if (max_abs > zeros_eps) {
          log_error(LogTest,
                    "Validation FAILED (zeros mode: max_abs > {:.1e})",
                    zeros_eps);
          pass = false;
        } else {
          log_info(LogTest, "Validation PASSED");
        }
      } else {
        // Comparison
        float pcc = 0.0f;
        float rmse = 0.0f;
        float relative_rmse = 0.0f;
        {
          ZoneScopedN("ComputeMM Host Validation Metrics");
          pcc = get_pcc(golden_vec, device_vec);
          rmse = get_rmse(golden_vec, device_vec);
          relative_rmse = get_relative_rmse(device_vec, golden_vec);
        }
        log_validation_sample_pairs("ComputeMM Validation", golden_vec,
                                    device_vec, 12);

        log_info(LogTest,
                 "Validation Result: PCC = {:.4f}, RMSE = {:.4f}, Relative RMSE "
                 "= {:.4f}",
                 pcc, rmse, relative_rmse);

        if (pcc < 0.99f) {
          log_error(LogTest, "Validation FAILED (PCC < 0.99)");
          pass = false;
        } else {
          log_info(LogTest, "Validation PASSED");
        }
      }
    }

  } catch (const std::exception &e) {
    pass = false;
    log_error(LogTest, "{}", e.what());
  }
  return pass;
}

bool test_compute_mm_async(tt::tt_metal::distributed::MeshDevice *device,
                     const TestParams &params) {
  bool pass = true;
  try {
    ZoneScopedN("ComputeMMAsync Functional Blocks");
    log_info(LogTest, "Starting Compute MM Test");
    log_info(LogTest, "M={}, N={}, K={}", params.M, params.N, params.K);
    log_info(LogTest, "Dispatch mode: async batch enqueue + single finish");

    auto arch = device->arch();
    uint32_t l1_size = 0;
    uint32_t l1_unreserved_base = 0;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    bool fp32_dest_acc_en = false;
    uint32_t Mt = 0, Nt = 0, Kt = 0;
    uint32_t num_cores_x = params.core_x;
    uint32_t num_cores_y = params.core_y;
    CoreCoord core_range(num_cores_x, num_cores_y);
    uint32_t per_core_Mt = 0;
    uint32_t per_core_Nt = 0;
    tt::DataFormat data_format = tt::DataFormat::Bfp8_b;
    uint32_t single_tile_size = 0;
    uint32_t out_block_h = 0, out_block_w = 0, in0_block_w = 0;
    uint32_t num_blocks_h = 1, num_blocks_w = 1;
    uint32_t out_subblock_h = 0, out_subblock_w = 0;
    uint32_t in0_cb_addr = 0, in1_cb_addr = 0, in2_cb_addr = 0, out_cb_addr = 0,
             interm_cb_addr = 0, in0_addr = 0, in1_addr = 0, out_addr = 0;
    BenchmarkInputs inputs;
    uint32_t effective_in0_addr = 0;
    uint32_t effective_in1_addr = 0;
    uint32_t effective_out_addr = 0;

    {
      ZoneScopedN("ComputeMMAsync Input Data Processing");

      {
        ZoneScopedN("ComputeMMAsync Host Setup and Blocking");
        l1_size = get_l1_size(arch);
        l1_unreserved_base =
            device->allocator()->get_base_allocator_addr(HalMemType::L1);
        std::tie(math_fidelity, fp32_dest_acc_en) = get_compute_params(arch);

        std::tie(Mt, Nt, Kt) =
            get_aligned_input_tile_num(params.M, params.N, params.K);

        per_core_Mt = ((Mt - 1) / num_cores_y) + 1;
        per_core_Nt = ((Nt - 1) / num_cores_x) + 1;

        data_format = compute_data_format_from_input_dtype(params.input_dtype);
        single_tile_size = tt::tile_size(data_format);

        if (params.use_dram) {
          auto [obh, obw, bw] = get_dynamic_l1_block_params(
              per_core_Mt, per_core_Nt, Kt, single_tile_size, l1_size,
              l1_unreserved_base);
          out_block_h = obh;
          out_block_w = obw;
          in0_block_w = bw;
          num_blocks_h = per_core_Mt / out_block_h;
          num_blocks_w = per_core_Nt / out_block_w;

          log_info(LogTest,
                   "DRAM blocking: per_core={}x{}, out_block={}x{}, "
                   "num_blocks={}x{}, in0_block_w={}",
                   per_core_Mt, per_core_Nt, out_block_h, out_block_w,
                   num_blocks_h, num_blocks_w, in0_block_w);
        } else {
          out_block_h = per_core_Mt;
          out_block_w = per_core_Nt;
          in0_block_w =
              get_in0_block_w(per_core_Mt, per_core_Nt, Kt, single_tile_size,
                              l1_size, l1_unreserved_base, false);
        }

        if (in0_block_w == 0) {
          uint32_t out_cb_tiles = out_block_h * out_block_w;
          uint32_t out_cb_bytes = out_cb_tiles * single_tile_size;
          uint32_t avail_l1 = l1_size - l1_unreserved_base;
          log_error(LogTest,
                    "Insufficient L1 memory for M={}, N={}, K={} "
                    "(out_block={}x{}, out_CB={}tiles={}KB, "
                    "avail_L1={}KB, dram={})",
                    params.M, params.N, params.K, out_block_h, out_block_w,
                    out_cb_tiles, out_cb_bytes / 1024, avail_l1 / 1024,
                    params.use_dram ? "yes" : "no");
          return false;
        }

        std::tie(out_subblock_h, out_subblock_w) =
            get_out_subblock_params(out_block_h, out_block_w);

        std::tie(in0_cb_addr, in1_cb_addr, in2_cb_addr, out_cb_addr,
                 interm_cb_addr, in0_addr, in1_addr, out_addr) =
            get_all_buffers_addresses(out_block_h, out_block_w, in0_block_w,
                                      single_tile_size, l1_unreserved_base,
                                      params.use_dram);
      }

      {
        ZoneScopedN("ComputeMMAsync Host Prepare Inputs");
        inputs = prepare_inputs_compute_mm(
            device, core_range, Mt, Nt, Kt, per_core_Mt, per_core_Nt,
            in0_block_w, single_tile_size, in0_addr, in1_addr, in2_cb_addr,
            params.use_dram, 0, data_format, params.pack_device,
          !params.bypass_check,
          params.unpack_device,
          params.zeros_mode);
      }

      {
        ZoneScopedN("ComputeMMAsync Host Resolve Buffer Addresses");
        effective_in0_addr = in0_addr;
        effective_in1_addr = in1_addr;
        effective_out_addr = out_addr;
        if (params.use_dram) {
          // MeshBuffer doesn't expose address() directly; extract the
          // underlying per-device Buffer* via get_device_buffer().
          auto coord = tt::tt_metal::distributed::MeshCoordinate(0, 0);
          effective_in0_addr =
              inputs.in0_buffer->get_device_buffer(coord)->address();
          effective_in1_addr =
              inputs.in1_buffer->get_device_buffer(coord)->address();
          effective_out_addr =
              inputs.out_buffer->get_device_buffer(coord)->address();
          log_info(LogTest, "DRAM mode: in0=0x{:x}, in1=0x{:x}, out=0x{:x}",
                   effective_in0_addr, effective_in1_addr, effective_out_addr);
        }
      }
    }

    // 6. Create Program and Run (single program — kernel handles multi-block)
    log_info(LogTest, "Num tests {}", params.num_iters);

    tt_metal::Program program;
    {
      ZoneScopedN("ComputeMMAsync Host Program Build");
      create_program_compute_mm(
          device, data_format, math_fidelity, fp32_dest_acc_en,
          single_tile_size, core_range, Mt, Nt, Kt, in0_block_w, out_subblock_h,
          out_subblock_w, per_core_Mt, per_core_Nt, out_block_h, out_block_w,
          num_blocks_h, num_blocks_w, in0_cb_addr, in1_cb_addr, in2_cb_addr,
          out_cb_addr, interm_cb_addr, effective_in0_addr, effective_in1_addr,
          effective_out_addr, params.use_dram, program);
    }

    auto mesh_workload = tt_metal::distributed::MeshWorkload();
    mesh_workload.add_program(
        tt::tt_metal::distributed::MeshCoordinateRange{{0, 0}, {0, 0}},
        std::move(program));

    {
      ZoneScopedN("ComputeMMAsync Host Dispatch");
      auto t_enqueue_begin = std::chrono::steady_clock::now();
      for (uint32_t i = 0; i < params.num_iters; ++i) {
        ZoneScopedN("ComputeMMAsync Host Dispatch Iteration");
        ZoneValue(i);
        {
          ZoneScopedN("ComputeMMAsync Host Enqueue");
          tt_metal::distributed::EnqueueMeshWorkload(
              device->mesh_command_queue(), mesh_workload, false);
        }
      }
      auto t_enqueue_end = std::chrono::steady_clock::now();

      auto t_finish_begin = std::chrono::steady_clock::now();
      {
        ZoneScopedN("ComputeMMAsync Host FinishWait");
        tt_metal::distributed::Finish(device->mesh_command_queue());
      }
      auto t_finish_end = std::chrono::steady_clock::now();

      auto enqueue_us =
          std::chrono::duration_cast<std::chrono::microseconds>(t_enqueue_end -
                                                                t_enqueue_begin)
              .count();
      auto finish_wait_us =
          std::chrono::duration_cast<std::chrono::microseconds>(t_finish_end -
                                                                t_finish_begin)
              .count();
      auto total_us =
          std::chrono::duration_cast<std::chrono::microseconds>(t_finish_end -
                                                                t_enqueue_begin)
              .count();
        const double async_per_op_exec_us =
          params.num_iters > 0
            ? static_cast<double>(total_us) /
              static_cast<double>(params.num_iters)
            : 0.0;
      log_info(LogTest,
               "Async batch timing: enqueue_window={}us, finish_wait={}us, "
               "total={}us, iters={}",
               enqueue_us, finish_wait_us, total_us, params.num_iters);
        log_info(
          LogTest,
          "Async cost model (X-ops): exec_total={}us, per_op_exec={:.3f}us, "
          "async_ops_executed={}, compare_ops_basis={}, "
          "host_setup_readback_excluded=true",
          total_us, async_per_op_exec_us, params.num_iters, params.num_iters);
    }

    if (!params.bypass_check) {
      ZoneScopedN("ComputeMMAsync Host Post Processing");
      log_info(LogTest, "Validation Started...");
      log_validation_pipeline_steps(params);

      if (params.use_dram && params.pack_device && !params.zeros_mode) {
        ZoneScopedN("ComputeMMAsync Host Validation: Reconstruct Effective Inputs");

        // Reconstruct effective packed inputs for rigorous golden-reference
        // validation. This aligns device-pack path with cpu-pack semantics.
        std::vector<uint32_t> in0_packed_effective;
        {
          ZoneScopedN("ComputeMMAsync Host Validation: Read Packed IN0 (Device Pack)");
          tt::tt_metal::distributed::ReadShard(
              device->mesh_command_queue(), in0_packed_effective,
              inputs.in0_buffer, tt::tt_metal::distributed::MeshCoordinate(0, 0),
              true);
        }
        {
          ZoneScopedN("ComputeMMAsync Host Validation: Decode Effective IN0 (Device Pack)");
          inputs.in0_vec = unpack_device_tiles_to_fp32(in0_packed_effective,
                                                       Mt * 32, Kt * 32,
                                                       data_format);
        }

        std::vector<uint32_t> in1_packed_effective;
        {
          ZoneScopedN("ComputeMMAsync Host Validation: Read Packed IN1 (Device Pack)");
          tt::tt_metal::distributed::ReadShard(
              device->mesh_command_queue(), in1_packed_effective,
              inputs.in1_buffer, tt::tt_metal::distributed::MeshCoordinate(0, 0),
              true);
        }
        {
          ZoneScopedN("ComputeMMAsync Host Validation: Decode Effective IN1 (Device Pack)");
          inputs.in1_vec = unpack_device_tiles_to_fp32(in1_packed_effective,
                                                       Kt * 32, Nt * 32,
                                                       data_format);
        }
      }

      // Compute Golden Reference (or trivially all zeros in zeros_mode).
      std::vector<float> golden_vec;
      {
        ZoneScopedN("ComputeMMAsync Host Golden Reference");
        if (params.zeros_mode) {
          log_info(LogTest, "Golden Reference: all-zeros (zeros mode)");
          golden_vec.assign(static_cast<size_t>(params.M) * params.N, 0.0f);
        } else {
          log_info(LogTest, "Computing Golden Reference (FP32)...");
          golden_vec = matmul_reference(inputs.in0_vec, inputs.in1_vec,
                                        params.M, params.N, params.K);
        }
      }

      // Read Back Results
      log_info(LogTest, "Reading Device Results...");
      std::vector<float> device_vec(params.M * params.N, 0.0f);
        const bool output_needs_trim =
          ((params.M % constants::TILE_HEIGHT) != 0) ||
          ((params.N % constants::TILE_WIDTH) != 0);
      auto *target_device = device->get_devices()[0];

      {
        ZoneScopedN("ComputeMMAsync Host Device Readback");
        if (params.use_dram) {
          if (params.unpack_device) {
            ZoneScopedN("ComputeMMAsync Host Device Unpack (ttnn)");
            if (!inputs.output_ttnn_tensor.has_value()) {
              throw std::runtime_error(
                  "Device unpack requested, but no output TTNN tensor was "
                  "created. Ensure output tensor bootstrap is enabled.");
            }
            auto &device_tensor_out = inputs.output_ttnn_tensor.value();

            if (params.output_dtype_mode == OutputDTypeMode::Fp32) {
              // 1. Typecast to FLOAT32 while still in TILE layout
              ttnn::Tensor device_tensor_fp32_tile;
              {
                ZoneScopedN("ttnn::typecast(FLOAT32)");
                device_tensor_fp32_tile =
                    ttnn::typecast(device_tensor_out, ttnn::DataType::FLOAT32);
              }

              // 2. Convert to ROW_MAJOR layout on device
              ttnn::Tensor device_tensor_fp32_rm;
              {
                ZoneScopedN("ttnn::untilize");
                device_tensor_fp32_rm = ttnn::untilize(device_tensor_fp32_tile);
              }

              {
                ZoneScopedN("ttnn::untilize_Finish");
                tt::tt_metal::distributed::Finish(device->mesh_command_queue());
              }

              // 3. Bring to host as vector<float>
              {
                ZoneScopedN("ttnn::to_vector<float> (Readback)");
                device_vec = device_tensor_fp32_rm.to_vector<float>();
              }
            } else {
              // Native/BF16 export path: untilize directly, no explicit FP32 cast.
              ttnn::Tensor device_tensor_rm;
              {
                ZoneScopedN("ttnn::untilize");
                device_tensor_rm = ttnn::untilize(device_tensor_out);
              }
              {
                ZoneScopedN("ttnn::untilize_Finish");
                tt::tt_metal::distributed::Finish(device->mesh_command_queue());
              }
              {
                ZoneScopedN("ttnn::to_vector<float> (Readback)");
                device_vec = device_tensor_rm.to_vector<float>();
              }
            }

            // Trim to actual M x N if needed
            if (output_needs_trim &&
                device_vec.size() > (size_t)(params.M * params.N)) {
              {
                ZoneScopedN("ComputeMMAsync Host Trim Device Readback to MxN");
                std::vector<float> trimmed(params.M * params.N, 0.0f);
                for (uint32_t r = 0; r < params.M; ++r) {
                  for (uint32_t c = 0; c < params.N; ++c) {
                    trimmed[r * params.N + c] = device_vec[r * (Nt * 32) + c];
                  }
                }
                device_vec = trimmed;
              }
            }
          } else {
            // === DRAM Readback ===
            // Read entire output buffer from DRAM
            std::vector<uint32_t> out_data;
            {
              ZoneScopedN("ComputeMMAsync Host ReadShard DRAM Output");
              tt::tt_metal::distributed::ReadShard(
                  device->mesh_command_queue(), out_data, inputs.out_buffer,
                  tt::tt_metal::distributed::MeshCoordinate(0, 0), true);
            }

            {
              ZoneScopedN("ComputeMMAsync Host Decode DRAM");
              // Unpack and untilize the full output
              device_vec = unpack_device_tiles_to_fp32(out_data, Mt * 32,
                                                       Nt * 32, data_format);
              // Trim to actual M x N if needed
              if (output_needs_trim &&
                  device_vec.size() > (size_t)(params.M * params.N)) {
                {
                  ZoneScopedN("ComputeMMAsync Host Trim DRAM Decode to MxN");
                  std::vector<float> trimmed(params.M * params.N, 0.0f);
                  for (uint32_t r = 0; r < params.M; ++r) {
                    for (uint32_t c = 0; c < params.N; ++c) {
                      trimmed[r * params.N + c] = device_vec[r * (Nt * 32) + c];
                    }
                  }
                  device_vec = trimmed;
                }
              }
            }
          }
        } else {
          // === L1 Readback (per-core: read all raw tiles first) ===
          std::vector<std::vector<uint32_t>> all_core_tiles(num_cores_y *
                                                            num_cores_x);
          {
            ZoneScopedN("ComputeMMAsync Host ReadFromDeviceL1 All Cores");
            for (int y = 0; y < (int)num_cores_y; y++) {
              for (int x = 0; x < (int)num_cores_x; x++) {
                CoreCoord core = {(std::size_t)x, (std::size_t)y};
                uint32_t core_n_tiles = per_core_Mt * per_core_Nt;
                uint32_t read_size = core_n_tiles * single_tile_size;

                tt_metal::detail::ReadFromDeviceL1(
                    target_device, core, out_addr, read_size,
                    all_core_tiles[y * num_cores_x + x]);
              }
            }
          }

          {
            ZoneScopedN("ComputeMMAsync Host Decode L1");
            // Unpack & stitch all cores
            for (int y = 0; y < (int)num_cores_y; y++) {
              for (int x = 0; x < (int)num_cores_x; x++) {
                auto core_data_untilized = unpack_device_tiles_to_fp32(
                    all_core_tiles[y * num_cores_x + x], per_core_Mt * 32,
                    per_core_Nt * 32, data_format);

                uint32_t global_r_start = y * per_core_Mt * 32;
                uint32_t global_c_start = x * per_core_Nt * 32;
                uint32_t r_len = per_core_Mt * 32;
                uint32_t c_len = per_core_Nt * 32;

                for (uint32_t r = 0; r < r_len; ++r) {
                  for (uint32_t c = 0; c < c_len; ++c) {
                    uint32_t global_idx = (global_r_start + r) * (params.N) +
                                          (global_c_start + c);
                    if (global_idx < device_vec.size()) {
                      device_vec[global_idx] =
                          core_data_untilized[r * c_len + c];
                    }
                  }
                }
              }
            }
          }
        }
      }

      if (params.output_dtype_mode == OutputDTypeMode::Bf16) {
        ZoneScopedN("ComputeMMAsync Host Output DType Postprocess BF16");
        for (auto &v : device_vec) {
          v = static_cast<float>(bfloat16(v));
        }
      }

      if (params.zeros_mode) {
        float max_abs = 0.0f;
        {
          ZoneScopedN("ComputeMMAsync Host Validation Metrics");
          max_abs = get_max_abs(device_vec);
        }
        log_validation_sample_pairs("ComputeMMAsync Validation (zeros)",
                                    golden_vec, device_vec, 12);
        const float zeros_eps = 1e-6f;
        log_info(LogTest,
                 "Validation Result (zeros mode): max_abs={:.6e}, eps={:.1e}",
                 max_abs, zeros_eps);
        if (max_abs > zeros_eps) {
          log_error(LogTest,
                    "Validation FAILED (zeros mode: max_abs > {:.1e})",
                    zeros_eps);
          pass = false;
        } else {
          log_info(LogTest, "Validation PASSED");
        }
      } else {
        // Comparison
        float pcc = 0.0f;
        float rmse = 0.0f;
        float relative_rmse = 0.0f;
        {
          ZoneScopedN("ComputeMMAsync Host Validation Metrics");
          pcc = get_pcc(golden_vec, device_vec);
          rmse = get_rmse(golden_vec, device_vec);
          relative_rmse = get_relative_rmse(device_vec, golden_vec);
        }
        log_validation_sample_pairs("ComputeMMAsync Validation", golden_vec,
                                    device_vec, 12);

        log_info(LogTest,
                 "Validation Result: PCC = {:.4f}, RMSE = {:.4f}, Relative RMSE "
                 "= {:.4f}",
                 pcc, rmse, relative_rmse);

        if (pcc < 0.99f) {
          log_error(LogTest, "Validation FAILED (PCC < 0.99)");
          pass = false;
        } else {
          log_info(LogTest, "Validation PASSED");
        }
      }
    }

  } catch (const std::exception &e) {
    pass = false;
    log_error(LogTest, "{}", e.what());
  }
  return pass;
}


bool test_compute_mm_trace(tt::tt_metal::distributed::MeshDevice *device,
                           const TestParams &params) {
  bool pass = true;
  try {
    ZoneScopedN("ComputeMMTrace: ComputeMM Functional Blocks");
    log_info(LogTest, "Starting Compute MM Test");
    log_info(LogTest, "M={}, N={}, K={}", params.M, params.N, params.K);
    log_info(LogTest, "Dispatch mode: trace capture + replay");

    auto arch = device->arch();
    uint32_t l1_size = 0;
    uint32_t l1_unreserved_base = 0;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    bool fp32_dest_acc_en = false;
    uint32_t Mt = 0, Nt = 0, Kt = 0;
    uint32_t num_cores_x = params.core_x;
    uint32_t num_cores_y = params.core_y;
    CoreCoord core_range(num_cores_x, num_cores_y);
    uint32_t per_core_Mt = 0;
    uint32_t per_core_Nt = 0;
    tt::DataFormat data_format = tt::DataFormat::Bfp8_b;
    uint32_t single_tile_size = 0;
    uint32_t out_block_h = 0, out_block_w = 0, in0_block_w = 0;
    uint32_t num_blocks_h = 1, num_blocks_w = 1;
    uint32_t out_subblock_h = 0, out_subblock_w = 0;
    uint32_t in0_cb_addr = 0, in1_cb_addr = 0, in2_cb_addr = 0, out_cb_addr = 0,
             interm_cb_addr = 0, in0_addr = 0, in1_addr = 0, out_addr = 0;
    BenchmarkInputs inputs;
    uint32_t effective_in0_addr = 0;
    uint32_t effective_in1_addr = 0;
    uint32_t effective_out_addr = 0;

    {
      ZoneScopedN("ComputeMMTrace: ComputeMM Input Data Processing");

      {
        ZoneScopedN("ComputeMMTrace: ComputeMM Host Setup and Blocking");
        l1_size = get_l1_size(arch);
        l1_unreserved_base =
            device->allocator()->get_base_allocator_addr(HalMemType::L1);
        std::tie(math_fidelity, fp32_dest_acc_en) = get_compute_params(arch);

        std::tie(Mt, Nt, Kt) =
            get_aligned_input_tile_num(params.M, params.N, params.K);

        per_core_Mt = ((Mt - 1) / num_cores_y) + 1;
        per_core_Nt = ((Nt - 1) / num_cores_x) + 1;

        data_format = compute_data_format_from_input_dtype(params.input_dtype);
        single_tile_size = tt::tile_size(data_format);

        if (params.use_dram) {
          auto [obh, obw, bw] = get_dynamic_l1_block_params(
              per_core_Mt, per_core_Nt, Kt, single_tile_size, l1_size,
              l1_unreserved_base);
          out_block_h = obh;
          out_block_w = obw;
          in0_block_w = bw;
          num_blocks_h = per_core_Mt / out_block_h;
          num_blocks_w = per_core_Nt / out_block_w;

          log_info(LogTest,
                   "DRAM blocking: per_core={}x{}, out_block={}x{}, "
                   "num_blocks={}x{}, in0_block_w={}",
                   per_core_Mt, per_core_Nt, out_block_h, out_block_w,
                   num_blocks_h, num_blocks_w, in0_block_w);
        } else {
          out_block_h = per_core_Mt;
          out_block_w = per_core_Nt;
          in0_block_w =
              get_in0_block_w(per_core_Mt, per_core_Nt, Kt, single_tile_size,
                              l1_size, l1_unreserved_base, false);
        }

        if (in0_block_w == 0) {
          uint32_t out_cb_tiles = out_block_h * out_block_w;
          uint32_t out_cb_bytes = out_cb_tiles * single_tile_size;
          uint32_t avail_l1 = l1_size - l1_unreserved_base;
          log_error(LogTest,
                    "Insufficient L1 memory for M={}, N={}, K={} "
                    "(out_block={}x{}, out_CB={}tiles={}KB, "
                    "avail_L1={}KB, dram={})",
                    params.M, params.N, params.K, out_block_h, out_block_w,
                    out_cb_tiles, out_cb_bytes / 1024, avail_l1 / 1024,
                    params.use_dram ? "yes" : "no");
          return false;
        }

        std::tie(out_subblock_h, out_subblock_w) =
            get_out_subblock_params(out_block_h, out_block_w);

        std::tie(in0_cb_addr, in1_cb_addr, in2_cb_addr, out_cb_addr,
                 interm_cb_addr, in0_addr, in1_addr, out_addr) =
            get_all_buffers_addresses(out_block_h, out_block_w, in0_block_w,
                                      single_tile_size, l1_unreserved_base,
                                      params.use_dram);
      }

      {
        ZoneScopedN("ComputeMMTrace: ComputeMM Host Prepare Inputs");
        inputs = prepare_inputs_compute_mm(
            device, core_range, Mt, Nt, Kt, per_core_Mt, per_core_Nt,
            in0_block_w, single_tile_size, in0_addr, in1_addr, in2_cb_addr,
            params.use_dram, 0, data_format, params.pack_device,
          !params.bypass_check,
          params.unpack_device,
          params.zeros_mode);
      }

      {
        ZoneScopedN("ComputeMMTrace: ComputeMM Host Resolve Buffer Addresses");
        effective_in0_addr = in0_addr;
        effective_in1_addr = in1_addr;
        effective_out_addr = out_addr;
        if (params.use_dram) {
          // MeshBuffer doesn't expose address() directly; extract the
          // underlying per-device Buffer* via get_device_buffer().
          auto coord = tt::tt_metal::distributed::MeshCoordinate(0, 0);
          effective_in0_addr =
              inputs.in0_buffer->get_device_buffer(coord)->address();
          effective_in1_addr =
              inputs.in1_buffer->get_device_buffer(coord)->address();
          effective_out_addr =
              inputs.out_buffer->get_device_buffer(coord)->address();
          log_info(LogTest, "DRAM mode: in0=0x{:x}, in1=0x{:x}, out=0x{:x}",
                   effective_in0_addr, effective_in1_addr, effective_out_addr);
        }
      }
    }

    // 6. Create Program and Run (single program — kernel handles multi-block)
    log_info(LogTest, "Num tests {}", params.num_iters);

    tt_metal::Program program;
    {
      ZoneScopedN("ComputeMMTrace: ComputeMM Host Program Build");
      create_program_compute_mm(
          device, data_format, math_fidelity, fp32_dest_acc_en,
          single_tile_size, core_range, Mt, Nt, Kt, in0_block_w, out_subblock_h,
          out_subblock_w, per_core_Mt, per_core_Nt, out_block_h, out_block_w,
          num_blocks_h, num_blocks_w, in0_cb_addr, in1_cb_addr, in2_cb_addr,
          out_cb_addr, interm_cb_addr, effective_in0_addr, effective_in1_addr,
          effective_out_addr, params.use_dram, program);
    }

    auto mesh_workload = tt_metal::distributed::MeshWorkload();
    mesh_workload.add_program(
        tt::tt_metal::distributed::MeshCoordinateRange{{0, 0}, {0, 0}},
        std::move(program));

    {
      ZoneScopedN("ComputeMMTrace: ComputeMM Host Dispatch");
      const uint8_t trace_cq_id = 0;
      auto &trace_cq = device->mesh_command_queue(trace_cq_id);
      uint32_t ops_per_capture = params.trace_capture_ops;
      if (ops_per_capture == 0) {
        ops_per_capture = 1;
      }

      const uint64_t replay_ops_total =
          static_cast<uint64_t>(params.num_iters) *
          static_cast<uint64_t>(ops_per_capture);
      const uint64_t compare_ops_basis = static_cast<uint64_t>(params.num_iters);

      log_info(LogTest,
               "Trace replay config: capture_ops_per_trace={}, replay_iters={}, "
               "replayed_ops_total={}, compare_ops_basis={}",
               ops_per_capture, params.num_iters, replay_ops_total,
               compare_ops_basis);
      if (ops_per_capture != 1) {
        log_warning(
            LogTest,
            "capture_ops_per_trace={} means replayed_ops_total={} for "
            "compare_ops_basis={}; for strict side-by-side fairness with async "
            "num_iters, set --trace-capture-ops 1.",
            ops_per_capture, replay_ops_total, compare_ops_basis);
      }

      std::optional<tt::tt_metal::distributed::MeshTraceId> trace_id;
      bool capture_ended = false;
      int64_t capture_record_us = -1;
      try {
        // Warmup once so compile/setup work is not included in trace replay.
        {
          ZoneScopedN("ComputeMMTrace: ComputeMM Trace Warmup");
          tt_metal::distributed::EnqueueMeshWorkload(
              trace_cq, mesh_workload, false);
          tt_metal::distributed::Finish(trace_cq);
        }

        {
          ZoneScopedN("ComputeMMTrace: ComputeMM Trace Capture");
          auto t_capture_begin = std::chrono::steady_clock::now();
          trace_id = tt_metal::distributed::BeginTraceCapture(device, trace_cq_id);
          for (uint32_t op = 0; op < ops_per_capture; ++op) {
            tt_metal::distributed::EnqueueMeshWorkload(
                trace_cq, mesh_workload, false);
          }
          device->end_mesh_trace(trace_cq_id, trace_id.value());
          auto t_capture_end = std::chrono::steady_clock::now();
            capture_record_us =
              std::chrono::duration_cast<std::chrono::microseconds>(
                  t_capture_end - t_capture_begin)
                  .count();
          log_info(LogTest,
                   "Trace timing: capture_record_window={}us, capture_ops={}",
                   capture_record_us, ops_per_capture);
          capture_ended = true;
        }

        {
          ZoneScopedN("ComputeMMTrace: ComputeMM Trace Replay Loop");
          auto t_replay_issue_begin = std::chrono::steady_clock::now();
          for (uint32_t i = 0; i < params.num_iters; ++i) {
            ZoneScopedN("ComputeMMTrace: ComputeMM Trace Replay Iteration");
            ZoneValue(i);
            device->replay_mesh_trace(trace_cq_id, trace_id.value(), false);
          }
          auto t_replay_issue_end = std::chrono::steady_clock::now();

          auto t_replay_finish_begin = std::chrono::steady_clock::now();
          tt_metal::distributed::Finish(trace_cq);
          auto t_replay_finish_end = std::chrono::steady_clock::now();

          auto replay_issue_us =
              std::chrono::duration_cast<std::chrono::microseconds>(
                  t_replay_issue_end - t_replay_issue_begin)
                  .count();
          auto replay_finish_wait_us =
              std::chrono::duration_cast<std::chrono::microseconds>(
                  t_replay_finish_end - t_replay_finish_begin)
                  .count();
          auto replay_total_us =
              std::chrono::duration_cast<std::chrono::microseconds>(
                  t_replay_finish_end - t_replay_issue_begin)
                  .count();
            const uint64_t total_cost_for_x_us =
              static_cast<uint64_t>(replay_total_us) +
              static_cast<uint64_t>(std::max<int64_t>(capture_record_us, 0));
            const double trace_per_op_exec_us =
              replay_ops_total > 0
                ? static_cast<double>(replay_total_us) /
                  static_cast<double>(replay_ops_total)
                : 0.0;
            const double trace_per_op_total_us =
              replay_ops_total > 0
                ? static_cast<double>(total_cost_for_x_us) /
                  static_cast<double>(replay_ops_total)
                : 0.0;
          log_info(LogTest,
                   "Trace timing: replay_issue_window={}us, finish_wait={}us, "
                   "replay_total={}us, replay_iters={}",
                   replay_issue_us, replay_finish_wait_us, replay_total_us,
                   params.num_iters);
            log_info(
              LogTest,
              "Trace cost model (X-ops): setup_capture={}us, exec_replay={}us, "
              "total_for_replayed_ops={}us, per_op_exec={:.3f}us, "
              "per_op_total_with_capture={:.3f}us, trace_ops_captured={}, "
              "trace_ops_replayed={}, compare_ops_basis={}, "
              "host_setup_readback_excluded=true",
              std::max<int64_t>(capture_record_us, 0), replay_total_us,
              total_cost_for_x_us, trace_per_op_exec_us, trace_per_op_total_us,
              ops_per_capture, replay_ops_total, compare_ops_basis);
        }

        {
          ZoneScopedN("ComputeMMTrace: Trace Release");
          device->release_mesh_trace(trace_id.value());
        }
        trace_id.reset();
      } catch (...) {
        if (trace_id.has_value()) {
          if (!capture_ended) {
            try {
              ZoneScopedN("ComputeMMTrace: Trace Forced End (Exception)");
              device->end_mesh_trace(trace_cq_id, trace_id.value());
            } catch (...) {
            }
          }
          try {
            ZoneScopedN("ComputeMMTrace: Trace FinishWait (Exception)");
            tt_metal::distributed::Finish(trace_cq);
          } catch (...) {
          }
          try {
            ZoneScopedN("ComputeMMTrace: Trace Release (Exception)");
            device->release_mesh_trace(trace_id.value());
          } catch (...) {
          }
        }
        throw;
      }
    }

    if (!params.bypass_check) {
      ZoneScopedN("ComputeMMTrace: ComputeMM Host Post Processing");
      log_info(LogTest, "Validation Started...");
      log_validation_pipeline_steps(params);

      if (params.use_dram && params.pack_device && !params.zeros_mode) {
        ZoneScopedN("ComputeMMTrace: ComputeMM Host Validation: Reconstruct Effective Inputs");

        // Reconstruct effective packed inputs for rigorous golden-reference
        // validation. This aligns device-pack path with cpu-pack semantics.
        std::vector<uint32_t> in0_packed_effective;
        {
          ZoneScopedN("ComputeMMTrace: ComputeMM Host Validation: Read Packed IN0 (Device Pack)");
          tt::tt_metal::distributed::ReadShard(
              device->mesh_command_queue(), in0_packed_effective,
              inputs.in0_buffer, tt::tt_metal::distributed::MeshCoordinate(0, 0),
              true);
        }
        {
          ZoneScopedN("ComputeMMTrace: ComputeMM Host Validation: Decode Effective IN0 (Device Pack)");
          inputs.in0_vec = unpack_device_tiles_to_fp32(in0_packed_effective,
                                                       Mt * 32, Kt * 32,
                                                       data_format);
        }

        std::vector<uint32_t> in1_packed_effective;
        {
          ZoneScopedN("ComputeMMTrace: ComputeMM Host Validation: Read Packed IN1 (Device Pack)");
          tt::tt_metal::distributed::ReadShard(
              device->mesh_command_queue(), in1_packed_effective,
              inputs.in1_buffer, tt::tt_metal::distributed::MeshCoordinate(0, 0),
              true);
        }
        {
          ZoneScopedN("ComputeMMTrace: ComputeMM Host Validation: Decode Effective IN1 (Device Pack)");
          inputs.in1_vec = unpack_device_tiles_to_fp32(in1_packed_effective,
                                                       Kt * 32, Nt * 32,
                                                       data_format);
        }
      }

      // Compute Golden Reference (or trivially all zeros in zeros_mode).
      std::vector<float> golden_vec;
      {
        ZoneScopedN("ComputeMMTrace: ComputeMM Host Golden Reference");
        if (params.zeros_mode) {
          log_info(LogTest, "Golden Reference: all-zeros (zeros mode)");
          golden_vec.assign(static_cast<size_t>(params.M) * params.N, 0.0f);
        } else {
          log_info(LogTest, "Computing Golden Reference (FP32)...");
          golden_vec = matmul_reference(inputs.in0_vec, inputs.in1_vec,
                                        params.M, params.N, params.K);
        }
      }

      // Read Back Results
      log_info(LogTest, "Reading Device Results...");
      std::vector<float> device_vec(params.M * params.N, 0.0f);
        const bool output_needs_trim =
          ((params.M % constants::TILE_HEIGHT) != 0) ||
          ((params.N % constants::TILE_WIDTH) != 0);
      auto *target_device = device->get_devices()[0];

      {
        ZoneScopedN("ComputeMMTrace: ComputeMM Host Device Readback");
        if (params.use_dram) {
          if (params.unpack_device) {
            ZoneScopedN("ComputeMMTrace: ComputeMM Host Device Unpack (ttnn)");
            if (!inputs.output_ttnn_tensor.has_value()) {
              throw std::runtime_error(
                  "Device unpack requested, but no output TTNN tensor was "
                  "created. Ensure output tensor bootstrap is enabled.");
            }
            auto &device_tensor_out = inputs.output_ttnn_tensor.value();

            if (params.output_dtype_mode == OutputDTypeMode::Fp32) {
              // 1. Typecast to FLOAT32 while still in TILE layout
              ttnn::Tensor device_tensor_fp32_tile;
              {
                ZoneScopedN("ComputeMMTrace: ttnn::typecast(FLOAT32)");
                device_tensor_fp32_tile =
                    ttnn::typecast(device_tensor_out, ttnn::DataType::FLOAT32);
              }

              // 2. Convert to ROW_MAJOR layout on device
              ttnn::Tensor device_tensor_fp32_rm;
              {
                ZoneScopedN("ComputeMMTrace: ttnn::untilize");
                device_tensor_fp32_rm = ttnn::untilize(device_tensor_fp32_tile);
              }

              {
                ZoneScopedN("ComputeMMTrace: ttnn::untilize_Finish");
                tt::tt_metal::distributed::Finish(device->mesh_command_queue());
              }

              // 3. Bring to host as vector<float>
              {
                ZoneScopedN("ComputeMMTrace: ttnn::to_vector<float> (Readback)");
                device_vec = device_tensor_fp32_rm.to_vector<float>();
              }
            } else {
              // Native/BF16 export path: untilize directly, no explicit FP32 cast.
              ttnn::Tensor device_tensor_rm;
              {
                ZoneScopedN("ComputeMMTrace: ttnn::untilize");
                device_tensor_rm = ttnn::untilize(device_tensor_out);
              }
              {
                ZoneScopedN("ComputeMMTrace: ttnn::untilize_Finish");
                tt::tt_metal::distributed::Finish(device->mesh_command_queue());
              }
              {
                ZoneScopedN("ComputeMMTrace: ttnn::to_vector<float> (Readback)");
                device_vec = device_tensor_rm.to_vector<float>();
              }
            }

            // Trim to actual M x N if needed
            if (output_needs_trim &&
                device_vec.size() > (size_t)(params.M * params.N)) {
              {
                ZoneScopedN("ComputeMMTrace: ComputeMM Host Trim Device Readback to MxN");
                std::vector<float> trimmed(params.M * params.N, 0.0f);
                for (uint32_t r = 0; r < params.M; ++r) {
                  for (uint32_t c = 0; c < params.N; ++c) {
                    trimmed[r * params.N + c] = device_vec[r * (Nt * 32) + c];
                  }
                }
                device_vec = trimmed;
              }
            }
          } else {
            // === DRAM Readback ===
            // Read entire output buffer from DRAM
            std::vector<uint32_t> out_data;
            {
              ZoneScopedN("ComputeMMTrace: ComputeMM Host ReadShard DRAM Output");
              tt::tt_metal::distributed::ReadShard(
                  device->mesh_command_queue(), out_data, inputs.out_buffer,
                  tt::tt_metal::distributed::MeshCoordinate(0, 0), true);
            }

            {
              ZoneScopedN("ComputeMMTrace: ComputeMM Host Decode DRAM");
              // Unpack and untilize the full output
              device_vec = unpack_device_tiles_to_fp32(out_data, Mt * 32,
                                                       Nt * 32, data_format);
              // Trim to actual M x N if needed
              if (output_needs_trim &&
                  device_vec.size() > (size_t)(params.M * params.N)) {
                {
                  ZoneScopedN("ComputeMMTrace: ComputeMM Host Trim DRAM Decode to MxN");
                  std::vector<float> trimmed(params.M * params.N, 0.0f);
                  for (uint32_t r = 0; r < params.M; ++r) {
                    for (uint32_t c = 0; c < params.N; ++c) {
                      trimmed[r * params.N + c] = device_vec[r * (Nt * 32) + c];
                    }
                  }
                  device_vec = trimmed;
                }
              }
            }
          }
        } else {
          // === L1 Readback (per-core: read all raw tiles first) ===
          std::vector<std::vector<uint32_t>> all_core_tiles(num_cores_y *
                                                            num_cores_x);
          {
            ZoneScopedN("ComputeMMTrace: ComputeMM Host ReadFromDeviceL1 All Cores");
            for (int y = 0; y < (int)num_cores_y; y++) {
              for (int x = 0; x < (int)num_cores_x; x++) {
                CoreCoord core = {(std::size_t)x, (std::size_t)y};
                uint32_t core_n_tiles = per_core_Mt * per_core_Nt;
                uint32_t read_size = core_n_tiles * single_tile_size;

                tt_metal::detail::ReadFromDeviceL1(
                    target_device, core, out_addr, read_size,
                    all_core_tiles[y * num_cores_x + x]);
              }
            }
          }

          {
            ZoneScopedN("ComputeMMTrace: ComputeMM Host Decode L1");
            // Unpack & stitch all cores
            for (int y = 0; y < (int)num_cores_y; y++) {
              for (int x = 0; x < (int)num_cores_x; x++) {
                auto core_data_untilized = unpack_device_tiles_to_fp32(
                    all_core_tiles[y * num_cores_x + x], per_core_Mt * 32,
                    per_core_Nt * 32, data_format);

                uint32_t global_r_start = y * per_core_Mt * 32;
                uint32_t global_c_start = x * per_core_Nt * 32;
                uint32_t r_len = per_core_Mt * 32;
                uint32_t c_len = per_core_Nt * 32;

                for (uint32_t r = 0; r < r_len; ++r) {
                  for (uint32_t c = 0; c < c_len; ++c) {
                    uint32_t global_idx = (global_r_start + r) * (params.N) +
                                          (global_c_start + c);
                    if (global_idx < device_vec.size()) {
                      device_vec[global_idx] =
                          core_data_untilized[r * c_len + c];
                    }
                  }
                }
              }
            }
          }
        }
      }

      if (params.output_dtype_mode == OutputDTypeMode::Bf16) {
        ZoneScopedN("ComputeMMTrace: ComputeMM Host Output DType Postprocess BF16");
        for (auto &v : device_vec) {
          v = static_cast<float>(bfloat16(v));
        }
      }

      if (params.zeros_mode) {
        float max_abs = 0.0f;
        {
          ZoneScopedN("ComputeMMTrace: ComputeMM Host Validation Metrics");
          max_abs = get_max_abs(device_vec);
        }
        log_validation_sample_pairs("ComputeMMTrace Validation (zeros)",
                                    golden_vec, device_vec, 12);
        const float zeros_eps = 1e-6f;
        log_info(LogTest,
                 "Validation Result (zeros mode): max_abs={:.6e}, eps={:.1e}",
                 max_abs, zeros_eps);
        if (max_abs > zeros_eps) {
          log_error(LogTest,
                    "Validation FAILED (zeros mode: max_abs > {:.1e})",
                    zeros_eps);
          pass = false;
        } else {
          log_info(LogTest, "Validation PASSED");
        }
      } else {
        // Comparison
        float pcc = 0.0f;
        float rmse = 0.0f;
        float relative_rmse = 0.0f;
        {
          ZoneScopedN("ComputeMMTrace: ComputeMM Host Validation Metrics");
          pcc = get_pcc(golden_vec, device_vec);
          rmse = get_rmse(golden_vec, device_vec);
          relative_rmse = get_relative_rmse(device_vec, golden_vec);
        }
        log_validation_sample_pairs("ComputeMMTrace Validation", golden_vec,
                                    device_vec, 12);

        log_info(LogTest,
                 "Validation Result: PCC = {:.4f}, RMSE = {:.4f}, Relative RMSE "
                 "= {:.4f}",
                 pcc, rmse, relative_rmse);

        if (pcc < 0.99f) {
          log_error(LogTest, "Validation FAILED (PCC < 0.99)");
          pass = false;
        } else {
          log_info(LogTest, "Validation PASSED");
        }
      }
    }

  } catch (const std::exception &e) {
    pass = false;
    log_error(LogTest, "{}", e.what());
  }
  return pass;
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
  double avg_write_bytes_per_iter =
      static_cast<double>(s.bytes_written) / iters;
  double avg_read_bytes_per_iter = static_cast<double>(s.bytes_read) / iters;

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
           "{} host-only pipeline timing (avg per iteration, us): "
           "input_generation={:.2f}, tilize_pack_transform={:.2f}, "
           "h2d_write={:.2f}, "
           "d2h_read={:.2f}, unpack_untilize_inverse_transform={:.2f}, "
           "host_end_to_end={:.2f}",
           tag, avg_generate_us, avg_transform_us, avg_write_us, avg_read_us,
           avg_inverse_transform_us, avg_end_to_end_us);

  log_info(
      LogTest,
      "{} host-only transfer summary (all iterations): "
      "total_h2d_bytes={}, total_d2h_bytes={}, avg_h2d_bytes_per_iter={:.0f}, "
      "avg_d2h_bytes_per_iter={:.0f}, effective_h2d_bw={:.3f} GB/s, "
      "effective_d2h_bw={:.3f} GB/s",
      tag, s.bytes_written, s.bytes_read, avg_write_bytes_per_iter,
      avg_read_bytes_per_iter, write_gbps, read_gbps);
}

bool test_host_pipeline_compute_mm(
    tt::tt_metal::distributed::MeshDevice *device, const TestParams &params) {
  bool pass = true;
  try {
    ZoneScopedN("HostPipeline ComputeMM Functional Blocks");
    log_info(LogTest,
             "Starting Host-Only ComputeMM Pipeline Test (no kernel dispatch)");

    auto [Mt, Nt, Kt] =
        get_aligned_input_tile_num(params.M, params.N, params.K);

    if (params.num_iters == 0) {
      log_error(LogTest, "Host-only ComputeMM requires --num-iters > 0");
      return false;
    }

    tt::DataFormat pipeline_data_format =
      compute_data_format_from_input_dtype(params.input_dtype);

    uint32_t single_tile_size = tt::tile_size(pipeline_data_format);
    uint32_t in0_num_tiles = Mt * Kt;
    uint32_t in1_num_tiles = Kt * Nt;
    uint32_t in0_size_bytes = in0_num_tiles * single_tile_size;
    uint32_t in1_size_bytes = in1_num_tiles * single_tile_size;

    tt::tt_metal::distributed::DeviceLocalBufferConfig device_local_in0{
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM,
    };
    tt::tt_metal::distributed::ReplicatedBufferConfig global_in0{
        .size = in0_size_bytes};
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> in0_buffer;
    {
      ZoneScopedN("HostPipeline ComputeMM Create IN0 MeshBuffer");
      in0_buffer = tt::tt_metal::distributed::MeshBuffer::create(
        global_in0, device_local_in0, device);
    }

    tt::tt_metal::distributed::DeviceLocalBufferConfig device_local_in1{
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM,
    };
    tt::tt_metal::distributed::ReplicatedBufferConfig global_in1{
        .size = in1_size_bytes};
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> in1_buffer;
    {
      ZoneScopedN("HostPipeline ComputeMM Create IN1 MeshBuffer");
      in1_buffer = tt::tt_metal::distributed::MeshBuffer::create(
        global_in1, device_local_in1, device);
    }

    HostPipelineStats stats;
    uint32_t completed_iters = 0;

    {
      ZoneScopedN("HostPipeline ComputeMM Host Dispatch");
      for (uint32_t i = 0; i < params.num_iters; ++i) {
        ZoneScopedN("HostPipeline ComputeMM Iteration");
        ZoneValue(i);
        auto t_iter_start = std::chrono::steady_clock::now();

        std::vector<float> in0_vec;
        std::vector<float> in1_vec;
        {
          ZoneScopedN("HostPipeline ComputeMM Prepare Inputs");
          auto t0 = std::chrono::steady_clock::now();
          if (params.zeros_mode) {
            in0_vec.assign(static_cast<size_t>(Mt) * Kt * constants::TILE_HW,
                           0.0f);
            in1_vec.assign(static_cast<size_t>(Kt) * Nt * constants::TILE_HW,
                           0.0f);
          } else {
            in0_vec = generate_fp32_random(Mt * Kt * constants::TILE_HW, 5.0f);
            in1_vec = generate_fp32_random(Kt * Nt * constants::TILE_HW, 5.0f);
          }
          auto t1 = std::chrono::steady_clock::now();
          stats.generate_us +=
              std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                  .count();
        }

        std::vector<uint32_t> in0_packed;
        std::vector<uint32_t> in1_packed;
        {
          ZoneScopedN("HostPipeline ComputeMM Transform Inputs");
          auto t0 = std::chrono::steady_clock::now();
          in0_packed = pack_tilized_fp32_to_device_format(
              in0_vec, Mt * 32, Kt * 32, pipeline_data_format);
          in1_packed = pack_tilized_fp32_to_device_format(
              in1_vec, Kt * 32, Nt * 32, pipeline_data_format);
          auto t1 = std::chrono::steady_clock::now();
          stats.transform_us +=
              std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                  .count();
        }

        {
          ZoneScopedN("HostPipeline ComputeMM Host Enqueue");
          auto t0 = std::chrono::steady_clock::now();
          tt::tt_metal::distributed::EnqueueWriteMeshBuffer(
              device->mesh_command_queue(), in0_buffer, in0_packed, false);
          tt::tt_metal::distributed::EnqueueWriteMeshBuffer(
              device->mesh_command_queue(), in1_buffer, in1_packed, false);
          tt::tt_metal::distributed::Finish(device->mesh_command_queue());
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
          ZoneScopedN("HostPipeline ComputeMM Host FinishWait");
          auto t0 = std::chrono::steady_clock::now();
          tt::tt_metal::distributed::ReadShard(
              device->mesh_command_queue(), in0_readback, in0_buffer,
              tt::tt_metal::distributed::MeshCoordinate(0, 0), true);
          tt::tt_metal::distributed::ReadShard(
              device->mesh_command_queue(), in1_readback, in1_buffer,
              tt::tt_metal::distributed::MeshCoordinate(0, 0), true);
          auto t1 = std::chrono::steady_clock::now();
          stats.read_us +=
              std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                  .count();
          stats.bytes_read +=
              static_cast<uint64_t>(in0_size_bytes + in1_size_bytes);
        }

        auto t_iter_end = std::chrono::steady_clock::now();
        stats.end_to_end_us +=
            std::chrono::duration_cast<std::chrono::microseconds>(t_iter_end -
                                                                  t_iter_start)
                .count();
        completed_iters++;

        // We break validation out of the loop timing, executing it only on the
        // FIRST iteration
        if (i == 0 && !params.bypass_check) {
          ZoneScopedN("HostPipeline ComputeMM Validation");
          auto t0_post = std::chrono::steady_clock::now();
          auto in0_untilized = unpack_device_tiles_to_fp32(
              in0_readback, Mt * 32, Kt * 32, pipeline_data_format);
          auto in1_untilized = unpack_device_tiles_to_fp32(
              in1_readback, Kt * 32, Nt * 32, pipeline_data_format);
          auto t1_post = std::chrono::steady_clock::now();
          stats.inverse_transform_us +=
              std::chrono::duration_cast<std::chrono::microseconds>(t1_post -
                                                                    t0_post)
                  .count();

          if (params.zeros_mode) {
            float in0_max = get_max_abs(in0_untilized);
            float in1_max = get_max_abs(in1_untilized);
            log_validation_sample_pairs("HostPipeline ComputeMM IN0 (zeros)",
                                        in0_vec, in0_untilized, 12);
            log_validation_sample_pairs("HostPipeline ComputeMM IN1 (zeros)",
                                        in1_vec, in1_untilized, 12);
            const float zeros_eps = 1e-6f;
            if (in0_max > zeros_eps || in1_max > zeros_eps) {
              log_error(LogTest,
                        "Host-only ComputeMM zeros roundtrip failed: "
                        "in0_max={:.6e}, in1_max={:.6e}",
                        in0_max, in1_max);
              pass = false;
              break;
            }
          } else {
            float in0_pcc = get_pcc(in0_vec, in0_untilized);
            float in1_pcc = get_pcc(in1_vec, in1_untilized);
            log_validation_sample_pairs("HostPipeline ComputeMM IN0", in0_vec,
                                        in0_untilized, 12);
            log_validation_sample_pairs("HostPipeline ComputeMM IN1", in1_vec,
                                        in1_untilized, 12);

            if (in0_pcc < 0.99f || in1_pcc < 0.99f) {
              log_error(LogTest,
                        "Host-only ComputeMM roundtrip check failed: "
                        "in0_pcc={:.4f}, in1_pcc={:.4f}",
                        in0_pcc, in1_pcc);
              pass = false;
              break;
            }
          }
        }
      }
    }

    log_info(LogTest,
             "Host-only ComputeMM pipeline dims: Mt={}, Nt={}, Kt={}, "
             "completed_iters={}",
             Mt, Nt, Kt, completed_iters);
    log_host_pipeline_stats("Test 3 (Host-Only ComputeMM Pipeline)", stats,
                            completed_iters);
  } catch (const std::exception &e) {
    pass = false;
    log_error(LogTest, "{}", e.what());
  }
  return pass;
}

bool test_host_pipeline_empty_tensor(
    tt::tt_metal::distributed::MeshDevice *device, const TestParams &params) {
  bool pass = true;
  try {
    ZoneScopedN("HostPipeline Empty Functional Blocks");
    log_info(
        LogTest,
        "Starting Host-Only Empty Tensor Pipeline Test (no kernel dispatch)");

    auto [Mt, Nt, Kt] =
        get_aligned_input_tile_num(params.M, params.N, params.K);
    (void)Kt;

    if (params.num_iters == 0) {
      log_error(LogTest, "Host-only Empty requires --num-iters > 0");
      return false;
    }

    tt::DataFormat pipeline_data_format =
      compute_data_format_from_input_dtype(params.input_dtype);

    uint32_t single_tile_size = tt::tile_size(pipeline_data_format);
    uint32_t num_tiles = Mt * Nt;
    uint32_t tensor_size_bytes = num_tiles * single_tile_size;
    uint64_t expected_tensor_elems = static_cast<uint64_t>(Mt) *
                                     static_cast<uint64_t>(Nt) *
                                     static_cast<uint64_t>(constants::TILE_HW);

    // No target_device needed — test 4 uses MeshBuffer::create directly.
    tt::tt_metal::distributed::DeviceLocalBufferConfig device_local{
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM,
    };
    tt::tt_metal::distributed::ReplicatedBufferConfig global_buf{
        .size = tensor_size_bytes};
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> tensor_buffer;
    {
      ZoneScopedN("HostPipeline Empty Create MeshBuffer");
      tensor_buffer = tt::tt_metal::distributed::MeshBuffer::create(
        global_buf, device_local, device);
    }

    HostPipelineStats stats;
    double transfer_window_us = 0.0;
    uint64_t transfer_window_bytes = 0;
    uint32_t completed_iters = 0;

    {
      ZoneScopedN("HostPipeline Empty Host Dispatch");
      for (uint32_t i = 0; i < params.num_iters; ++i) {
        ZoneScopedN("HostPipeline Empty Iteration");
        ZoneValue(i);
        auto t_iter_start = std::chrono::steady_clock::now();

        std::vector<float> tensor_vec;
        {
          ZoneScopedN("HostPipeline Empty Prepare Inputs");
          auto t0 = std::chrono::steady_clock::now();
          if (params.zeros_mode) {
            tensor_vec.assign(static_cast<size_t>(Mt) * Nt * constants::TILE_HW,
                              0.0f);
          } else {
            tensor_vec =
                generate_fp32_random(Mt * Nt * constants::TILE_HW, 5.0f);
          }
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
          ZoneScopedN("HostPipeline Empty Transform Inputs");
          auto t0 = std::chrono::steady_clock::now();
          packed = pack_tilized_fp32_to_device_format(
              tensor_vec, Mt * 32, Nt * 32, pipeline_data_format);
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
          ZoneScopedN("HostPipeline Empty Host Enqueue");
          auto t0 = std::chrono::steady_clock::now();
          tt::tt_metal::distributed::EnqueueWriteMeshBuffer(
              device->mesh_command_queue(), tensor_buffer, packed, false);
          tt::tt_metal::distributed::Finish(device->mesh_command_queue());
          auto t1 = std::chrono::steady_clock::now();
          stats.write_us +=
              std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                  .count();
          stats.bytes_written += static_cast<uint64_t>(tensor_size_bytes);
        }

        std::vector<uint32_t> readback;
        {
          ZoneScopedN("HostPipeline Empty Host FinishWait");
          auto t0 = std::chrono::steady_clock::now();
          tt::tt_metal::distributed::ReadShard(
              device->mesh_command_queue(), readback, tensor_buffer,
              tt::tt_metal::distributed::MeshCoordinate(0, 0), true);
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
            std::chrono::duration_cast<std::chrono::microseconds>(
                t_transfer_end - t_transfer_start)
                .count();
        transfer_window_bytes += static_cast<uint64_t>(tensor_size_bytes) * 2;

        auto t_iter_end = std::chrono::steady_clock::now();
        stats.end_to_end_us +=
            std::chrono::duration_cast<std::chrono::microseconds>(t_iter_end -
                                                                  t_iter_start)
                .count();
        completed_iters++;

        // We break validation out of the loop timing, executing it only on the
        // FIRST iteration
        if (i == 0 && !params.bypass_check) {
          ZoneScopedN("HostPipeline Empty Validation Processing");
          auto t0_post = std::chrono::steady_clock::now();
          auto roundtrip = unpack_device_tiles_to_fp32(
              readback, Mt * 32, Nt * 32, pipeline_data_format);
          auto t1_post = std::chrono::steady_clock::now();
          stats.inverse_transform_us +=
              std::chrono::duration_cast<std::chrono::microseconds>(t1_post -
                                                                    t0_post)
                  .count();

          if (params.zeros_mode) {
            float max_abs = get_max_abs(roundtrip);
            log_validation_sample_pairs("HostPipeline Empty (zeros)",
                                        tensor_vec, roundtrip, 12);
            const float zeros_eps = 1e-6f;
            if (max_abs > zeros_eps) {
              log_error(LogTest,
                        "Host-only Empty zeros roundtrip failed: "
                        "max_abs={:.6e}",
                        max_abs);
              pass = false;
              break;
            }
          } else {
            float pcc = get_pcc(tensor_vec, roundtrip);
            log_validation_sample_pairs("HostPipeline Empty", tensor_vec,
                                        roundtrip, 12);

            if (pcc < 0.99f) {
              log_error(LogTest,
                        "Host-only Empty roundtrip check failed: "
                        "pcc={:.4f}",
                        pcc);
              pass = false;
              break;
            }
          }
        }
      }
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
    log_host_pipeline_stats("Test 4 (Host-Only Empty Tensor Pipeline)", stats,
                            completed_iters);
  } catch (const std::exception &e) {
    pass = false;
    log_error(LogTest, "{}", e.what());
  }
  return pass;
}

///////////////////////////////////////////////////////////////////////////////////////////
/// Empty Kernel Launch Test
///////////////////////////////////////////////////////////////////////////////////////////
bool test_empty_kernel_launch(tt::tt_metal::distributed::MeshDevice *device,
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
        CoreCoord end_core = {(std::size_t)params.core_x - 1,
                              (core_group_idx == params.core_groups - 1)
                                  ? (std::size_t)params.core_y - 1
                                  : ((params.core_y / params.core_groups) *
                                     (core_group_idx + 1)) -
                                        1};
        CoreRange group_of_cores(start_core, end_core);

        log_info(LogTest,
                 "Setting kernels for core group {}, cores ({},{}) ~ ({},{})",
                 core_group_idx, start_core.x, start_core.y, end_core.x,
                 end_core.y);

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
          resolve_kernel_source_path("empty_reader.cpp"),
            group_of_cores,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt_metal::NOC::RISCV_1_default,
                .compile_args = reader_compile_args});

        std::vector<uint32_t> writer_compile_args = {uint32_t(core_group_idx)};
        auto writer_kernel = tt_metal::CreateKernel(
            program,
          resolve_kernel_source_path("empty_writer.cpp"),
            group_of_cores,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = writer_compile_args});

        std::vector<uint32_t> compute_compile_args = {uint32_t(core_group_idx)};
        tt_metal::CreateKernel(
            program,
          resolve_kernel_source_path("empty_compute.cpp"),
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
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////
    auto mesh_workload = tt_metal::distributed::MeshWorkload();
    mesh_workload.add_program(
        tt::tt_metal::distributed::MeshCoordinateRange{{0, 0}, {0, 0}},
        std::move(program));

    // Explicitly compile the program to measure compile overhead.
    // auto t_compile_begin = std::chrono::steady_clock::now();
    // for (auto &[range, prog] : mesh_workload.get_programs()) {
    //    tt_metal::detail::CompileProgram(device->get_devices()[0], prog);
    //}
    // auto t_compile_end = std::chrono::steady_clock::now();
    // auto compile_time =
    //  std::chrono::duration_cast<std::chrono::microseconds>(t_compile_end -
    //                               t_compile_begin)
    //    .count();
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
          tt_metal::distributed::EnqueueMeshWorkload(
              device->mesh_command_queue(), mesh_workload, false);
        }
        {
          ZoneScopedN("EmptyKernel Host FinishWait");
          tt_metal::distributed::Finish(device->mesh_command_queue());
        }

        // auto t_end = std::chrono::steady_clock::now();
        // elapsed_us.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t_end
        // - t_begin).count()); log_info(LogTest, "Time elapsed for executing
        // empty kernels: {}us", elapsed_us[i]);
      }
    }

    // Calculate stats
    // std::sort(elapsed_us.begin(), elapsed_us.end());

    // Filter outliers if we have enough data (e.g. > 2 samples)
    // std::vector<unsigned long> filtered_elapsed_us;
    // if (elapsed_us.size() > 2) {
    //     // Exclude min and max
    //     filtered_elapsed_us.assign(elapsed_us.begin() + 1,
    //     elapsed_us.end() - 1);
    //} else {
    //     filtered_elapsed_us = elapsed_us;
    //}
    //
    // auto min_val = *std::min_element(filtered_elapsed_us.begin(),
    // filtered_elapsed_us.end()); auto max_val =
    // *std::max_element(filtered_elapsed_us.begin(),
    // filtered_elapsed_us.end()); auto sum_val =
    // std::accumulate(filtered_elapsed_us.begin(),
    // filtered_elapsed_us.end(), 0.0); auto avg_val = sum_val /
    // filtered_elapsed_us.size();
    //
    // double sum_sq_diff = 0.0;
    // for (const auto& val : filtered_elapsed_us) {
    //    double diff = val - avg_val;
    //    sum_sq_diff += diff * diff;
    //}
    // auto std_dev = std::sqrt(sum_sq_diff / filtered_elapsed_us.size());
    // log_info(LogTest, "Execution Stats (us) [Trimmed]: Min={}, Max={},
    // Avg={:.2f}, StdDev={:.2f}", min_val, max_val, avg_val, std_dev);

    //        pass &= device->close(); // THIS WAS THE ORIGINAL VERSION, NOW
    //        WE CLOSE THE DEVICE IN MAIN
  } catch (const std::exception &e) {
    pass = false;
    log_error(LogTest, "{}", e.what());
    log_error(LogTest, "System error message: {}", std::strerror(errno));
  }
  return pass;
}

// Pins the process to a range of CPUs to reduce jitter while allowing for
// internal parallelism (OpenMP, metal worker threads).
// Binding to a single core causes severe contention in parallel regions.
void pin_to_cpu(uint32_t start_cpu_id, uint32_t num_cores_to_pin) {
  cpu_set_t mask;
  CPU_ZERO(&mask);

  uint32_t total_cpus = sysconf(_SC_NPROCESSORS_ONLN);

  for (uint32_t i = 0; i < num_cores_to_pin; ++i) {
    uint32_t target_cpu = (start_cpu_id + i) % total_cpus;
    CPU_SET(target_cpu, &mask);
  }

  int result = sched_setaffinity(0, sizeof(mask), &mask);
  if (result == 0) {
    log_info(
        LogTest,
        "Successfully pinned to CPU range {}-{} (Requested start={}, range={}, "
        "total={})",
        start_cpu_id, (start_cpu_id + num_cores_to_pin - 1) % total_cpus,
        start_cpu_id, num_cores_to_pin, total_cpus);
  } else {
    log_warning(LogTest, "Failed to pin to CPU range: {}", strerror(errno));
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
// Main Benchmarking Test
//////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
  // Disable Tracy to avoid interference with measurements
  unsetenv("TT_METAL_DEVICE_PROFILER");
  unsetenv("TRACY_ENABLE");

  if (std::getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
    log_error(tt::LogTest, "Test not supported w/ slow dispatch, exiting");
  }

  // Legacy Definition of max cores in each dimension for the architecture
  // being tested Left to know the limits
  /*
      uint32_t max_x = 0;
      uint32_t max_y = 0;
      const char *arch_env = std::getenv("ARCH_NAME");
      const std::string arch = arch_env ? arch_env : std::string();

   if (arch == "grayskull") {
          max_x = 11; max_y = 8;
          log_info(tt::LogTest, "Configured core range for grayskull:
   max_x={}, max_y={}", max_x, max_y); } else if (arch == "wormhole_b0") {
          max_x = 7; max_y = 6;
          log_info(tt::LogTest, "Configured core range for wormhole_b0:
   max_x={}, max_y={}", max_x, max_y); } else if (arch == "blackhole") {
          max_x = 12; max_y = 9;
          log_info(tt::LogTest, "Configured core range for blackhole:
   max_x={}, max_y={}", max_x, max_y); } else { log_error(tt::LogTest,
   "Unknown or unset ARCH_NAME ('{}'). Set ARCH_NAME env var or pass
   explicit --x_size/--y_size.", arch); throw std::runtime_error("Unknown
   ARCH_NAME; please set ARCH_NAME environment variable");
      }
  */

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

    // Create mesh device. Trace replay requires non-zero trace memory region.
  int device_id = 0;
    const bool enable_trace_mode = params.test == TestType::ComputeMMTraceReplay;
    const size_t trace_region_size = enable_trace_mode ? (256 * 1024) : 0;
    const size_t num_command_queues = 1;
    device_params.device = tt_metal::distributed::MeshDevice::create_unit_mesh(
      device_id, DEFAULT_L1_SMALL_SIZE, trace_region_size, num_command_queues);
    if (enable_trace_mode) {
    log_info(LogTest,
         "Trace mode device config: trace_region_size={} bytes, num_cqs={}",
         trace_region_size, num_command_queues);
    }
  device_params.grid_coord =
      device_params.device->compute_with_storage_grid_size();

  // Definition of max cores in each dimension for the architecture being
  // tested
  uint32_t max_x = device_params.grid_coord.x;
  uint32_t max_y = device_params.grid_coord.y;

  if (params.cpu_id != 0xFFFFFFFF) {
    pin_to_cpu(params.cpu_id, params.cpu_range);
  }

  //// Print test summary
  log_info(LogTest, "Full Characterization Benchmarking Test");
  log_info(LogTest, "=======================================");
  log_effective_configuration(params, device_params, max_x, max_y);

  if (params.core_x > max_x || params.core_y > max_y) {
    log_error(
        tt::LogTest,
        "Requested core size ({},{}) exceeds max for architecture ({},{})",
        params.core_x, params.core_y, max_x, max_y);
    if (params.use_cache) {
      disable_persistent_kernel_cache_if_available();
    }
    device_params.device->close();
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
    pass = test_empty_kernel_launch(device_params.device.get(), params);
    break;
  case TestType::ComputeMM:
    pass = test_compute_mm(device_params.device.get(), params);
    break;
  case TestType::ComputeMMAsyncBatch:
    pass = test_compute_mm_async(device_params.device.get(), params);
    break;
  case TestType::ComputeMMTraceReplay:
    pass = test_compute_mm_trace(device_params.device.get(), params);
    break;
  case TestType::SubDeviceMM:
    pass = test_sub_device_manager_mm(device_params.device.get(), params);
    break;
  case TestType::HostPipelineComputeMM:
    pass = test_host_pipeline_compute_mm(device_params.device.get(), params);
    break;
  case TestType::HostPipelineEmpty:
    pass = test_host_pipeline_empty_tensor(device_params.device.get(), params);
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
  device_params.device->close();

  return 0;
}
