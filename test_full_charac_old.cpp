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
#if __has_include(<tt-metalium/test_common.hpp>)
#include <tt-metalium/test_common.hpp>
#elif __has_include("tt_metal/api/tt-metalium/test_common.hpp")
#include "tt_metal/api/tt-metalium/test_common.hpp"
#elif __has_include("test_common.hpp")
#include "test_common.hpp"
#endif
#include <tt-metalium/tt_metal.hpp>
#if __has_include(<tt-metalium/tilize_utils.hpp>)
#include <tt-metalium/tilize_utils.hpp>
#define HAS_SWIZZLED_TILIZE_UTILS 1
#elif __has_include(<tt-metalium/tilize_untilize.hpp>)
#include <tt-metalium/tilize_untilize.hpp>
#define HAS_SWIZZLED_TILIZE_UTILS 0
#elif __has_include(<tt-metalium/test_tiles.hpp>)
#include <tt-metalium/test_tiles.hpp>
#define HAS_SWIZZLED_TILIZE_UTILS 0
#else
#define HAS_SWIZZLED_TILIZE_UTILS 0
#endif
#include <tt-metalium/persistent_kernel_cache.hpp>

// Utility / test helpers
// #include "test_common.hpp"
#include "hostdevcommon/common_values.hpp"
#include "hostdevcommon/kernel_structs.h"
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

void log_validation_sample_pairs(const std::string &tag,
                                 const std::vector<float> &reference,
                                 const std::vector<float> &observed,
                                 size_t sample_count = 12) {
  size_t common_size = std::min(reference.size(), observed.size());
  size_t num_samples = std::min(sample_count, common_size);
  log_info(LogTest,
           "{} visual sample check (showing {} of {} aligned elements)",
           tag, num_samples, common_size);

  for (size_t i = 0; i < num_samples; ++i) {
    float abs_err = std::fabs(reference[i] - observed[i]);
    log_info(LogTest, "{} [{}] ref={:.6f}, obs={:.6f}, abs_err={:.6f}", tag,
             i, reference[i], observed[i], abs_err);
  }
}

} // namespace

////////////////////////////////////////////////////////////////////////////////////////
// Default values for benchmark parameters
////////////////////////////////////////////////////////////////////////////////////////

constexpr uint32_t DEFAULT_ITERATIONS = 10000;
constexpr uint32_t MAX_ARGS = 255;
constexpr uint32_t L1_SAFETY_MARGIN_BYTES = 64 * 1024;

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
    vec[i] = rand_float();
  }
  return vec;
}

template <typename T>
std::vector<T> tilize_compat(const std::vector<T> &input, uint32_t rows,
                             uint32_t cols) {
  if (rows % constants::TILE_HEIGHT != 0 || cols % constants::TILE_WIDTH != 0) {
    throw std::runtime_error("tilize_compat requires rows/cols multiples of 32");
  }
  size_t expected = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  if (input.size() != expected) {
    throw std::runtime_error(
        "tilize_compat size mismatch: expected=" + std::to_string(expected) +
        ", got=" + std::to_string(input.size()));
  }

  std::vector<T> output;
  output.reserve(expected);
  for (uint32_t tr = 0; tr < rows; tr += constants::TILE_HEIGHT) {
    for (uint32_t tc = 0; tc < cols; tc += constants::TILE_WIDTH) {
      for (uint32_t r = 0; r < constants::TILE_HEIGHT; ++r) {
        for (uint32_t c = 0; c < constants::TILE_WIDTH; ++c) {
          size_t src = static_cast<size_t>(tr + r) * cols + (tc + c);
          output.push_back(input[src]);
        }
      }
    }
  }
  return output;
}

template <typename T>
std::vector<T> untilize_compat(const std::vector<T> &input, uint32_t rows,
                               uint32_t cols) {
  if (rows % constants::TILE_HEIGHT != 0 || cols % constants::TILE_WIDTH != 0) {
    throw std::runtime_error("untilize_compat requires rows/cols multiples of 32");
  }
  size_t expected = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  if (input.size() != expected) {
    throw std::runtime_error(
        "untilize_compat size mismatch: expected=" + std::to_string(expected) +
        ", got=" + std::to_string(input.size()));
  }

  std::vector<T> output(expected);
  size_t src = 0;
  for (uint32_t tr = 0; tr < rows; tr += constants::TILE_HEIGHT) {
    for (uint32_t tc = 0; tc < cols; tc += constants::TILE_WIDTH) {
      for (uint32_t r = 0; r < constants::TILE_HEIGHT; ++r) {
        for (uint32_t c = 0; c < constants::TILE_WIDTH; ++c) {
          size_t dst = static_cast<size_t>(tr + r) * cols + (tc + c);
          output[dst] = input[src++];
        }
      }
    }
  }
  return output;
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
  double avg_write_bytes_per_iter = static_cast<double>(s.bytes_written) / iters;
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
           "input_generation={:.2f}, tilize_pack_transform={:.2f}, h2d_write={:.2f}, "
           "d2h_read={:.2f}, unpack_untilize_inverse_transform={:.2f}, host_end_to_end={:.2f}",
           tag, avg_generate_us, avg_transform_us, avg_write_us, avg_read_us,
           avg_inverse_transform_us, avg_end_to_end_us);

  log_info(LogTest,
           "{} host-only transfer summary (all iterations): "
           "total_h2d_bytes={}, total_d2h_bytes={}, avg_h2d_bytes_per_iter={:.0f}, "
           "avg_d2h_bytes_per_iter={:.0f}, effective_h2d_bw={:.3f} GB/s, effective_d2h_bw={:.3f} GB/s",
           tag, s.bytes_written, s.bytes_read, avg_write_bytes_per_iter,
           avg_read_bytes_per_iter, write_gbps, read_gbps);
}

uint32_t get_l1_size_legacy(tt::ARCH arch) {
  constexpr uint32_t GS_L1_SIZE = 1048576;
  constexpr uint32_t WH_L1_SIZE = 1499136;
  constexpr uint32_t BH_L1_SIZE = 1499136;

  if (arch == tt::ARCH::WORMHOLE_B0) {
    return WH_L1_SIZE;
  }
  if (arch == tt::ARCH::GRAYSKULL) {
    return GS_L1_SIZE;
  }
  if (arch == tt::ARCH::BLACKHOLE) {
    return BH_L1_SIZE;
  }
  return 0;
}

std::tuple<MathFidelity, bool> get_compute_params_legacy(tt::ARCH arch) {
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

uint32_t get_in0_block_w_legacy(uint32_t per_core_Mt, uint32_t per_core_Nt,
                                uint32_t Kt, uint32_t single_tile_size,
                                uint32_t l1_size, uint32_t l1_unreserved_base,
                                bool use_dram = false) {
  std::vector<uint32_t> in0_block_w_choices = {4, 2, 1};
  uint32_t num_buffer = 2;
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

    uint32_t total_l1_needed =
        in0_cb_size + in1_cb_size + in2_cb_size + intermediate_cb_size +
        out_cb_size;
    if (!use_dram) {
      uint32_t per_core_in0_size = per_core_Mt * choice * single_tile_size;
      uint32_t per_core_in1_size = per_core_Nt * choice * single_tile_size;
      uint32_t per_core_out_size = per_core_Mt * per_core_Nt * single_tile_size;
      total_l1_needed += per_core_in0_size + per_core_in1_size + per_core_out_size;
    }

    if (l1_unreserved_base + total_l1_needed + L1_SAFETY_MARGIN_BYTES <=
        l1_size) {
      return choice;
    }
  }
  return 0;
}

std::tuple<uint32_t, uint32_t> get_out_subblock_params_legacy(
  uint32_t per_core_Mt, uint32_t per_core_Nt, uint32_t choice = 0) {
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

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
           uint32_t>
get_all_buffers_addresses_legacy(uint32_t per_core_Mt, uint32_t per_core_Nt,
                                 uint32_t in0_block_w,
                                 uint32_t single_tile_size,
                                 uint32_t l1_unreserved_base,
                                 bool use_dram = false) {
  uint32_t num_buffer = 2;
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

std::tuple<uint32_t, uint32_t, uint32_t>
get_multi_dim_per_core_factor_legacy(uint32_t per_core_M, uint32_t per_core_N,
                                      uint32_t Kt,
                                      uint32_t single_tile_size,
                                      uint32_t l1_size,
                                      uint32_t l1_unreserved_base) {
  constexpr uint32_t MAX_K_BLOCKS_PREFERRED = 16;

  uint32_t raw_avail_l1 = l1_size - l1_unreserved_base;
  uint32_t avail_l1 =
      (raw_avail_l1 > L1_SAFETY_MARGIN_BYTES)
          ? (raw_avail_l1 - L1_SAFETY_MARGIN_BYTES)
          : raw_avail_l1;

  auto fits_l1 = [&](uint32_t obh, uint32_t obw, uint32_t bw) -> bool {
    uint32_t in0_cb = obh * bw * 2 * single_tile_size;
    uint32_t in1_cb = obw * bw * 2 * single_tile_size;
    uint32_t in2_cb = 2 * single_tile_size;
    uint32_t out_cb = obh * obw * single_tile_size;
    uint32_t interm = obh * obw * single_tile_size;
    return (in0_cb + in1_cb + in2_cb + out_cb + interm) <= avail_l1;
  };

  if (fits_l1(per_core_M, per_core_N, Kt)) {
    return {per_core_M, per_core_N, Kt};
  }

  auto get_divisors = [](uint32_t n) -> std::vector<uint32_t> {
    std::vector<uint32_t> divs;
    for (uint32_t i = 1; i <= n; i++) {
      if (n % i == 0)
        divs.push_back(i);
    }
    return divs;
  };

  auto m_factors = get_divisors(per_core_M);
  auto n_factors = get_divisors(per_core_N);

  struct Pair {
    uint32_t m, n;
    uint32_t product;
    float ratio;
  };
  std::vector<Pair> pairs;
  for (auto mf : m_factors) {
    for (auto nf : n_factors) {
      float mx = (float)std::max(mf, nf);
      float mn = (float)std::min(mf, nf);
      pairs.push_back({mf, nf, mf * nf, mx / mn});
    }
  }
  std::sort(pairs.begin(), pairs.end(), [](const Pair &a, const Pair &b) {
    if (a.product != b.product)
      return a.product > b.product;
    return a.ratio < b.ratio;
  });

  std::vector<uint32_t> k_block_w_candidates;
  for (uint32_t bw = Kt; bw >= 1; bw--) {
    if (Kt % bw == 0) {
      k_block_w_candidates.push_back(bw);
    }
  }

  for (auto bw : k_block_w_candidates) {
    if ((Kt / bw) > MAX_K_BLOCKS_PREFERRED) {
      continue;
    }
    for (auto &p : pairs) {
      if (p.n != per_core_N) {
        continue;
      }
      if (fits_l1(p.m, p.n, bw)) {
        return {p.m, p.n, bw};
      }
    }
  }

  for (auto bw : k_block_w_candidates) {
    if ((Kt / bw) > MAX_K_BLOCKS_PREFERRED) {
      continue;
    }
    for (auto &p : pairs) {
      if (fits_l1(p.m, p.n, bw)) {
        return {p.m, p.n, bw};
      }
    }
  }

  for (auto &p : pairs) {
    for (auto bw : k_block_w_candidates) {
      if (fits_l1(p.m, p.n, bw)) {
        return {p.m, p.n, bw};
      }
    }
  }

  return {1, 1, 1};
}

template <typename T>
std::vector<T> get_row_slice(const std::vector<T> &data, int start_row_index,
                             int num_rows, int /*rows*/, int cols) {
  size_t start = static_cast<size_t>(start_row_index) * static_cast<size_t>(cols);
  size_t end =
      static_cast<size_t>(start_row_index + num_rows) * static_cast<size_t>(cols);
  if (end > data.size()) {
    throw std::runtime_error(
        "get_row_slice out of range: start=" + std::to_string(start) +
        ", end=" + std::to_string(end) +
        ", size=" + std::to_string(data.size()));
  }

  std::vector<T> result;
  result.reserve(end - start);
  for (size_t i = start; i < end; ++i) {
    result.push_back(data[i]);
  }
  return result;
}

template <typename T>
std::vector<T> get_col_slice(const std::vector<T> &data, int start_col_index,
                             int num_cols, int rows, int cols) {
  if (start_col_index < 0 || num_cols < 0 || rows < 0 || cols < 0 ||
      start_col_index + num_cols > cols) {
    throw std::runtime_error(
        "get_col_slice invalid arguments: start_col_index=" +
        std::to_string(start_col_index) + ", num_cols=" +
        std::to_string(num_cols) + ", rows=" + std::to_string(rows) +
        ", cols=" + std::to_string(cols));
  }
  size_t required_size = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  if (required_size > data.size()) {
    throw std::runtime_error(
        "get_col_slice input too small: required=" +
        std::to_string(required_size) + ", size=" +
        std::to_string(data.size()));
  }

  std::vector<T> result;
  result.reserve(static_cast<size_t>(rows) * static_cast<size_t>(num_cols));
  for (int r = 0; r < rows; r++) {
    for (int c = start_col_index; c < (start_col_index + num_cols); c++) {
      size_t idx = static_cast<size_t>(r) * static_cast<size_t>(cols) +
                   static_cast<size_t>(c);
      result.push_back(data[idx]);
    }
  }
  return result;
}

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
    return std::numeric_limits<float>::quiet_NaN();
  }

  double rrmse = std::sqrt(err_sq_sum / ref_sq_sum);
  return static_cast<float>(rrmse);
}

std::vector<float> matmul_reference(const std::vector<float> &a,
                                    const std::vector<float> &b, uint32_t M,
                                    uint32_t N, uint32_t K) {
  std::vector<float> c(M * N, 0.0f);
  for (uint32_t i = 0; i < M; ++i) {
    for (uint32_t k = 0; k < K; ++k) {
      double val_a = static_cast<double>(a[i * K + k]);
      for (uint32_t j = 0; j < N; ++j) {
        double current_val = static_cast<double>(c[i * N + j]);
        current_val += val_a * static_cast<double>(b[k * N + j]);
        c[i * N + j] = static_cast<float>(current_val);
      }
    }
  }
  return c;
}

struct BenchmarkInputsLegacy {
  std::vector<float> in0_vec;
  std::vector<float> in1_vec;
  std::shared_ptr<tt_metal::Buffer> in0_buffer;
  std::shared_ptr<tt_metal::Buffer> in1_buffer;
  std::shared_ptr<tt_metal::Buffer> out_buffer;
};

BenchmarkInputsLegacy prepare_inputs_compute_mm_legacy(
    tt::tt_metal::IDevice *device, CoreCoord core_range, uint32_t Mt,
    uint32_t Nt, uint32_t Kt, uint32_t per_core_Mt, uint32_t per_core_Nt,
    uint32_t in0_block_w, uint32_t single_tile_size, uint32_t in0_addr,
    uint32_t in1_addr, uint32_t in2_cb_addr, bool use_dram) {
  BenchmarkInputsLegacy inputs;
  inputs.in0_vec = generate_fp32_random(Mt * Kt * constants::TILE_HW, 5.0f);
  inputs.in1_vec = generate_fp32_random(Nt * Kt * constants::TILE_HW, 5.0f);
  std::vector<uint32_t> in2(single_tile_size / sizeof(uint32_t), 0);

  if (use_dram) {
    std::vector<float> in0_tilized;
    std::vector<uint32_t> in0_packed;
    try {
      in0_tilized = tilize_compat(inputs.in0_vec, Mt * 32, Kt * 32);
      in0_packed =
          pack_fp32_vec_as_bfp8_tiles(in0_tilized, /*row_major_input=*/true,
                                      /*is_exp_a=*/false);
    } catch (const std::exception &e) {
      throw std::runtime_error(
          "Legacy ComputeMM DRAM IN0 transform failed: Mt=" +
          std::to_string(Mt) + ", Kt=" + std::to_string(Kt) +
          ", in0_vec_size=" + std::to_string(inputs.in0_vec.size()) +
          ", what=" + e.what());
    }

    uint32_t in0_num_tiles = Mt * Kt;
    uint32_t in0_size_bytes = in0_num_tiles * single_tile_size;
    inputs.in0_buffer = tt_metal::CreateBuffer(tt_metal::InterleavedBufferConfig{
        .device = device,
        .size = in0_size_bytes,
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM});
    tt_metal::detail::WriteToBuffer(inputs.in0_buffer, in0_packed);

    try {
      auto in0_unpacked = unpack_bfp8_tiles_into_float_vec(
          in0_packed, /*row_major_output=*/true, /*is_exp_a=*/false);
      inputs.in0_vec = untilize_compat(in0_unpacked, Mt * 32, Kt * 32);
    } catch (const std::exception &e) {
      throw std::runtime_error(
          "Legacy ComputeMM DRAM IN0 inverse-transform failed: Mt=" +
          std::to_string(Mt) + ", Kt=" + std::to_string(Kt) +
          ", what=" + e.what());
    }

    std::vector<float> in1_tilized;
    std::vector<uint32_t> in1_packed;
    try {
      in1_tilized = tilize_compat(inputs.in1_vec, Kt * 32, Nt * 32);
      in1_packed =
          pack_fp32_vec_as_bfp8_tiles(in1_tilized, /*row_major_input=*/true,
                                      /*is_exp_a=*/false);
    } catch (const std::exception &e) {
      throw std::runtime_error(
          "Legacy ComputeMM DRAM IN1 transform failed: Kt=" +
          std::to_string(Kt) + ", Nt=" + std::to_string(Nt) +
          ", in1_vec_size=" + std::to_string(inputs.in1_vec.size()) +
          ", what=" + e.what());
    }
    try {
      auto in1_unpacked = unpack_bfp8_tiles_into_float_vec(
          in1_packed, /*row_major_output=*/true, /*is_exp_a=*/false);
      inputs.in1_vec = untilize_compat(in1_unpacked, Kt * 32, Nt * 32);
    } catch (const std::exception &e) {
      throw std::runtime_error(
          "Legacy ComputeMM DRAM IN1 inverse-transform failed: Kt=" +
          std::to_string(Kt) + ", Nt=" + std::to_string(Nt) +
          ", what=" + e.what());
    }

    uint32_t in1_num_tiles = Kt * Nt;
    uint32_t in1_size_bytes = in1_num_tiles * single_tile_size;
    inputs.in1_buffer = tt_metal::CreateBuffer(tt_metal::InterleavedBufferConfig{
        .device = device,
        .size = in1_size_bytes,
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM});
    tt_metal::detail::WriteToBuffer(inputs.in1_buffer, in1_packed);

    uint32_t out_num_tiles = Mt * Nt;
    uint32_t out_size_bytes = out_num_tiles * single_tile_size;
    inputs.out_buffer = tt_metal::CreateBuffer(tt_metal::InterleavedBufferConfig{
        .device = device,
        .size = out_size_bytes,
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM});

    for (uint32_t y = 0; y < core_range.y; y++) {
      for (uint32_t x = 0; x < core_range.x; x++) {
        CoreCoord core = {(std::size_t)x, (std::size_t)y};
        tt_metal::detail::WriteToDeviceL1(device, core, in2_cb_addr, in2);
      }
    }
  } else {
    std::fill(inputs.in1_vec.begin(), inputs.in1_vec.end(), 0.0f);
    uint32_t rows = Kt * 32;
    uint32_t cols = Nt * 32;
    uint32_t diag = std::min(rows, cols);
    for (uint32_t i = 0; i < diag; ++i) {
      inputs.in1_vec[i * cols + i] = 1.0f;
    }

    uint32_t last_block_h =
        Mt % per_core_Mt == 0 ? per_core_Mt : Mt % per_core_Mt;
    uint32_t last_block_w =
        Nt % per_core_Nt == 0 ? per_core_Nt : Nt % per_core_Nt;

    for (int r = 0; r < (int)core_range.y; r++) {
      int num_r = (r == (int)core_range.y - 1) ? (last_block_h) : (per_core_Mt);
      std::vector<float> in0_slice =
          get_row_slice(inputs.in0_vec, r * per_core_Mt * 32, num_r * 32,
                        Mt * 32, Kt * 32);
      auto in0_block_slice =
          get_col_slice(in0_slice, 0, in0_block_w * 32, num_r * 32, Kt * 32);
          std::vector<float> in0_block_tilized;
          std::vector<uint32_t> in0;
          try {
          in0_block_tilized =
            tilize_compat(in0_block_slice, num_r * 32, in0_block_w * 32);
          in0 = pack_fp32_vec_as_bfp8_tiles(in0_block_tilized,
                            /*row_major_input=*/true,
                            /*is_exp_a=*/false);
          } catch (const std::exception &e) {
          throw std::runtime_error(
            "Legacy ComputeMM L1 IN0 transform failed: num_r=" +
            std::to_string(num_r) + ", in0_block_w=" +
            std::to_string(in0_block_w) + ", Kt=" + std::to_string(Kt) +
            ", in0_block_slice_size=" + std::to_string(in0_block_slice.size()) +
            ", what=" + e.what());
          }

      for (int c = 0; c < (int)core_range.x; c++) {
        int num_c =
            (c == (int)core_range.x - 1) ? (last_block_w) : (per_core_Nt);

        std::vector<float> in1_block_slice(in0_block_w * num_c * 1024,
                                           (float)0);
        int num_ones = std::min(in0_block_w, static_cast<uint32_t>(num_c)) * 32;
        for (int i = 0; i < num_ones; i++) {
          size_t idx = static_cast<size_t>(i) *
                           static_cast<size_t>(num_c * 32) +
                       static_cast<size_t>(i);
          if (idx >= in1_block_slice.size()) {
            throw std::runtime_error(
                "in1 identity index out of range: idx=" +
                std::to_string(idx) + ", size=" +
                std::to_string(in1_block_slice.size()) +
                ", num_ones=" + std::to_string(num_ones) +
                ", num_c=" + std::to_string(num_c) +
                ", in0_block_w=" + std::to_string(in0_block_w));
          }
          in1_block_slice[idx] = (float)1;
        }

        std::vector<float> in1_block_tilized;
        std::vector<uint32_t> in1;
        try {
            in1_block_tilized =
              tilize_compat(in1_block_slice, in0_block_w * 32, num_c * 32);
          in1 = pack_fp32_vec_as_bfp8_tiles(in1_block_tilized,
                          /*row_major_input=*/true,
                          /*is_exp_a=*/false);
        } catch (const std::exception &e) {
          throw std::runtime_error(
            "Legacy ComputeMM L1 IN1 transform failed: in0_block_w=" +
            std::to_string(in0_block_w) + ", num_c=" +
            std::to_string(num_c) + ", in1_block_slice_size=" +
            std::to_string(in1_block_slice.size()) + ", what=" + e.what());
        }

        CoreCoord core = {(std::size_t)c, (std::size_t)r};
        tt_metal::detail::WriteToDeviceL1(device, core, in0_addr, in0);
        tt_metal::detail::WriteToDeviceL1(device, core, in1_addr, in1);
        tt_metal::detail::WriteToDeviceL1(device, core, in2_cb_addr, in2);
      }
    }
  }

  return inputs;
}

void create_program_compute_mm_legacy(
    tt::tt_metal::IDevice *device, tt::DataFormat cb_data_format,
    MathFidelity math_fidelity, bool fp32_dest_acc_en,
    uint32_t single_tile_size, CoreCoord core_range, uint32_t Mt, uint32_t Nt,
    uint32_t Kt, uint32_t in0_block_w, uint32_t out_subblock_h,
    uint32_t out_subblock_w, uint32_t per_core_Mt, uint32_t per_core_Nt,
    uint32_t out_block_h, uint32_t out_block_w, uint32_t num_blocks_h,
    uint32_t num_blocks_w, uint32_t in0_cb_addr, uint32_t in1_cb_addr,
    uint32_t in2_cb_addr, uint32_t out_cb_addr, uint32_t interm_cb_addr,
    uint32_t in0_addr, uint32_t in1_addr, uint32_t out_addr, bool use_dram,
    tt_metal::Program &program) {
  uint32_t num_buffer = 2;
  uint32_t in0_block_tiles = out_block_h * in0_block_w;
  uint32_t in0_CB_tiles = in0_block_tiles * num_buffer;
  uint32_t in1_block_tiles = out_block_w * in0_block_w;
  uint32_t in1_CB_tiles = in1_block_tiles * num_buffer;
  uint32_t out_block_tiles_count = out_block_h * out_block_w;
  uint32_t out_CB_tiles = out_block_tiles_count;
  uint32_t out_CB_size = out_CB_tiles * single_tile_size;

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

  std::vector<uint32_t> compute_kernel_args = {
      in0_block_w,            in0_num_subblocks,
      in0_block_num_tiles,    in0_subblock_num_tiles,
      in1_num_subblocks,      in1_block_num_tiles,
      in1_per_core_w,         num_blocks,
      out_subblock_h,         out_subblock_w,
      out_subblock_num_tiles, 1,
      num_blocks_w,           num_blocks_h};

  CoreRange all_cores({0, 0}, {(std::size_t)core_range.x - 1,
                               (std::size_t)core_range.y - 1});

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

  std::map<uint8_t, tt::DataFormat> cb_out_config_map = {
      {(uint8_t)tt::CBIndex::c_16, cb_data_format},
      {(uint8_t)tt::CBIndex::c_24, cb_data_format}};
  tt_metal::CircularBufferConfig cb_out_config =
      tt_metal::CircularBufferConfig(out_CB_size, cb_out_config_map)
          .set_page_size(tt::CBIndex::c_16, single_tile_size)
          .set_page_size(tt::CBIndex::c_24, single_tile_size);
  tt_metal::CreateCircularBuffer(program, CoreRangeSet({all_cores}),
                                 cb_out_config);

  std::string reader_kernel_path;
  std::string writer_kernel_path;
  std::map<std::string, std::string> writer_defines;

  if (use_dram) {
    reader_kernel_path = "tests/tt_metal/tt_metal/perf_microbenchmark/"
                         "13_full_charac/kernels/"
                         "in0_reader_bmm_tile_layout_dram.cpp";
    writer_kernel_path = "tests/tt_metal/tt_metal/perf_microbenchmark/"
                         "13_full_charac/kernels/"
                         "in1_reader_writer_bmm_tile_layout_dram.cpp";
  } else {
    reader_kernel_path = "tests/tt_metal/tt_metal/perf_microbenchmark/"
                         "13_full_charac/kernels/"
                         "in0_reader_bmm_tile_layout.cpp";
    writer_kernel_path = "tests/tt_metal/tt_metal/perf_microbenchmark/"
                         "13_full_charac/kernels/"
                         "in1_reader_writer_bmm_tile_layout.cpp";
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

  tt_metal::CreateKernel(
      program,
      "tests/tt_metal/tt_metal/perf_microbenchmark/13_full_charac/kernels/"
      "bmm_large_block_zm_fused_bias_activation.cpp",
      all_cores,
      tt_metal::ComputeConfig{.math_fidelity = math_fidelity,
                              .fp32_dest_acc_en = fp32_dest_acc_en,
                              .compile_args = compute_kernel_args});

  uint32_t last_block_h =
      Mt % per_core_Mt == 0 ? per_core_Mt : Mt % per_core_Mt;
  uint32_t last_block_w =
      Nt % per_core_Nt == 0 ? per_core_Nt : Nt % per_core_Nt;

  for (int y = 0; y < (int)core_range.y; y++) {
    for (int x = 0; x < (int)core_range.x; x++) {
      CoreCoord core = {(std::size_t)x, (std::size_t)y};
      auto phy_core = device->worker_core_from_logical_core(core);
      uint32_t cur_core_valid_Mt =
          (y == (int)core_range.y - 1) ? last_block_h : per_core_Mt;
      uint32_t cur_core_valid_Nt =
          (x == (int)core_range.x - 1) ? last_block_w : per_core_Nt;

      uint32_t last_outer_block_h =
          (cur_core_valid_Mt % out_block_h == 0) ? out_block_h
                                                  : (cur_core_valid_Mt % out_block_h);
      uint32_t last_outer_block_w =
          (cur_core_valid_Nt % out_block_w == 0) ? out_block_w
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
        uint32_t in0_start_tile_id = y * per_core_Mt * Kt;
        reader_args = {
            in0_addr,
            in0_start_tile_id,
            1,
            Kt,
            in0_block_w,
            in0_block_w,
            out_block_h,
            in0_block_w * out_block_h,
            num_blocks,
            (uint32_t)phy_core.x,
            (uint32_t)phy_core.y,
            last_outer_block_h,
            num_blocks_h,
            num_blocks_w,
            out_block_h * Kt,
            in2_cb_addr,
        };

        uint32_t in1_start_tile_id = x * per_core_Nt;
        uint32_t out_start_tile_id = y * per_core_Mt * Nt + x * per_core_Nt;

        writer_args = {
            in1_addr,
            in1_start_tile_id,
            1,
            Nt,
            in0_block_w * Nt,
            out_block_w,
            in0_block_w,
            out_block_w * in0_block_w,
            num_blocks,
            in2_cb_addr,
            (uint32_t)phy_core.x,
            (uint32_t)phy_core.y,
            out_addr,
            out_start_tile_id,
            1,
            Nt,
            out_subblock_w,
            out_subblock_h * Nt,
            out_subblock_w,
            out_subblock_h,
            out_subblock_num_tiles,
            in1_num_subblocks,
            in0_num_subblocks,
            last_outer_block_w,
            last_outer_num_nonzero_subblocks_h,
            last_outer_subblock_h,
            last_outer_padded_block_tiles_h_skip,
            last_outer_num_nonzero_subblocks_w,
            last_outer_subblock_w,
            last_outer_padded_subblock_tiles_addr_skip,
            last_outer_padded_block_tiles_w_skip,
            num_blocks_h,
            num_blocks_w,
            out_block_w,
            out_block_h * Nt,
            out_block_w,
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
              ((num_blocks_w - 1) * out_block_w) +
              ((out_block_h - 1) * Nt) + (out_block_w - 1);

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
        reader_args = {in0_addr,
                       0,
                       1,
                       in0_block_w,
                       in0_block_w,
                       in0_block_w,
                       per_core_Mt,
                       in0_block_w * per_core_Mt,
                       num_blocks,
                       (uint32_t)phy_core.x,
                       (uint32_t)phy_core.y,
                       cur_core_valid_Mt};

        writer_args = {in1_addr,
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
                       out_subblock_num_tiles,
                       per_core_Nt / out_subblock_w,
                       per_core_Mt / out_subblock_h,
                       cur_core_valid_Nt,
                       (y == (int)core_range.y - 1)
                           ? last_outer_num_nonzero_subblocks_h
                           : (per_core_Mt / out_subblock_h),
                       (y == (int)core_range.y - 1) ? last_outer_subblock_h
                                                     : out_subblock_h,
                       (y == (int)core_range.y - 1)
                           ? last_outer_padded_block_tiles_h_skip
                           : 0,
                       (x == (int)core_range.x - 1)
                           ? last_outer_num_nonzero_subblocks_w
                           : (per_core_Nt / out_subblock_w),
                       (x == (int)core_range.x - 1) ? last_outer_subblock_w
                                                     : out_subblock_w,
                       (x == (int)core_range.x - 1)
                           ? last_outer_padded_subblock_tiles_addr_skip
                           : 0,
                       (x == (int)core_range.x - 1)
                           ? last_outer_padded_block_tiles_w_skip
                           : 0};
      }

      tt_metal::SetRuntimeArgs(program, mm_reader_id, core, reader_args);
      tt_metal::SetRuntimeArgs(program, mm_writer_id, core, writer_args);
    }
  }
}

bool test_compute_mm(tt::tt_metal::IDevice *device, const TestParams &params) {
  bool pass = true;
  try {
    ZoneScopedN("ComputeMM Functional Blocks");
    log_info(LogTest, "Starting Compute MM Test");
    log_info(LogTest, "M={}, N={}, K={}", params.M, params.N, params.K);

    if (params.dtype != 0) {
      log_error(LogTest,
                "Legacy ComputeMM currently supports only --dtype 0 (BFP8), "
                "requested dtype={}",
                params.dtype);
      return false;
    }

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
    BenchmarkInputsLegacy inputs;
    uint32_t effective_in0_addr = 0;
    uint32_t effective_in1_addr = 0;
    uint32_t effective_out_addr = 0;

    {
      ZoneScopedN("ComputeMM Input Data Processing");

      {
        ZoneScopedN("ComputeMM Host Setup and Blocking");
        l1_size = get_l1_size_legacy(arch);
        l1_unreserved_base =
            device->allocator()->get_base_allocator_addr(HalMemType::L1);
        std::tie(math_fidelity, fp32_dest_acc_en) = get_compute_params_legacy(arch);

        std::tie(Mt, Nt, Kt) =
            get_aligned_input_tile_num(params.M, params.N, params.K);

        per_core_Mt = ((Mt - 1) / num_cores_y) + 1;
        per_core_Nt = ((Nt - 1) / num_cores_x) + 1;

        data_format = (params.dtype == 0) ? tt::DataFormat::Bfp8_b
                                          : tt::DataFormat::Float16_b;
        single_tile_size = tt::tile_size(data_format);

        if (params.use_dram) {
          auto [obh, obw, bw] = get_multi_dim_per_core_factor_legacy(
              per_core_Mt, per_core_Nt, Kt, single_tile_size, l1_size,
              l1_unreserved_base);
          out_block_h = obh;
          out_block_w = obw;
          in0_block_w = bw;
          num_blocks_h = per_core_Mt / out_block_h;
          num_blocks_w = per_core_Nt / out_block_w;

          log_info(LogTest,
                   "DRAM blocking: per_core={}x{}, out_block={}x{}, "
                   "num_blocks={}x{}, in0_block_w={}, safety_margin={}KB",
                   per_core_Mt, per_core_Nt, out_block_h, out_block_w,
                   num_blocks_h, num_blocks_w, in0_block_w,
                   L1_SAFETY_MARGIN_BYTES / 1024);
        } else {
          out_block_h = per_core_Mt;
          out_block_w = per_core_Nt;
          in0_block_w =
              get_in0_block_w_legacy(per_core_Mt, per_core_Nt, Kt,
                                     single_tile_size, l1_size,
                                     l1_unreserved_base, false);
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
            get_out_subblock_params_legacy(out_block_h, out_block_w);

        std::tie(in0_cb_addr, in1_cb_addr, in2_cb_addr, out_cb_addr,
                 interm_cb_addr, in0_addr, in1_addr, out_addr) =
            get_all_buffers_addresses_legacy(out_block_h, out_block_w,
                                             in0_block_w, single_tile_size,
                                             l1_unreserved_base,
                                             params.use_dram);
      }

      {
        ZoneScopedN("ComputeMM Host Prepare Inputs");
        inputs = prepare_inputs_compute_mm_legacy(
            device, core_range, Mt, Nt, Kt, per_core_Mt, per_core_Nt,
            in0_block_w, single_tile_size, in0_addr, in1_addr, in2_cb_addr,
            params.use_dram);
      }

      {
        ZoneScopedN("ComputeMM Host Resolve Buffer Addresses");
        effective_in0_addr = in0_addr;
        effective_in1_addr = in1_addr;
        effective_out_addr = out_addr;
        if (params.use_dram) {
          effective_in0_addr = inputs.in0_buffer->address();
          effective_in1_addr = inputs.in1_buffer->address();
          effective_out_addr = inputs.out_buffer->address();
          log_info(LogTest, "DRAM mode: in0=0x{:x}, in1=0x{:x}, out=0x{:x}",
                   effective_in0_addr, effective_in1_addr,
                   effective_out_addr);
        }
      }
    }

    log_info(LogTest, "Num tests {}", params.num_iters);

    tt_metal::Program program;
    {
      ZoneScopedN("ComputeMM Host Program Build");
      create_program_compute_mm_legacy(
          device, data_format, math_fidelity, fp32_dest_acc_en,
          single_tile_size, core_range, Mt, Nt, Kt, in0_block_w,
          out_subblock_h, out_subblock_w, per_core_Mt, per_core_Nt,
          out_block_h, out_block_w, num_blocks_h, num_blocks_w, in0_cb_addr,
          in1_cb_addr, in2_cb_addr, out_cb_addr, interm_cb_addr,
          effective_in0_addr, effective_in1_addr, effective_out_addr,
          params.use_dram, program);
    }

    {
      ZoneScopedN("ComputeMM Host Dispatch");
      for (uint32_t i = 0; i < params.num_iters; ++i) {
        ZoneScopedN("ComputeMM Host Dispatch Iteration");
        ZoneValue(i);
        {
          ZoneScopedN("ComputeMM Host Enqueue");
          EnqueueProgram(device->command_queue(), program, false);
        }
        {
          ZoneScopedN("ComputeMM Host FinishWait");
          Finish(device->command_queue());
        }
      }
    }

    if (!params.bypass_check) {
      ZoneScopedN("ComputeMM Host Post Processing");
      log_info(LogTest, "Validation Started...");

      std::vector<float> golden_vec;
      {
        ZoneScopedN("ComputeMM Host Golden Reference");
        log_info(LogTest, "Computing Golden Reference (FP32)...");
        golden_vec = matmul_reference(inputs.in0_vec, inputs.in1_vec, params.M,
                                      params.N, params.K);
      }

      log_info(LogTest, "Reading Device Results...");
      std::vector<float> device_vec(params.M * params.N, 0.0f);
      {
        ZoneScopedN("ComputeMM Host Device Readback and Decode");
        if (params.use_dram) {
          std::vector<uint32_t> out_data;
          tt_metal::detail::ReadFromBuffer(inputs.out_buffer, out_data);
          auto out_float = unpack_bfp8_tiles_into_float_vec(
              out_data, /*row_major_output=*/true, /*is_exp_a=*/false);
          device_vec = untilize_compat(out_float, Mt * 32, Nt * 32);
          if (device_vec.size() > (size_t)(params.M * params.N)) {
            std::vector<float> trimmed(params.M * params.N, 0.0f);
            for (uint32_t r = 0; r < params.M; ++r) {
              for (uint32_t c = 0; c < params.N; ++c) {
                trimmed[r * params.N + c] = device_vec[r * (Nt * 32) + c];
              }
            }
            device_vec = trimmed;
          }
        } else {
          for (int y = 0; y < (int)num_cores_y; y++) {
            for (int x = 0; x < (int)num_cores_x; x++) {
              CoreCoord core = {(std::size_t)x, (std::size_t)y};
              uint32_t core_n_tiles = per_core_Mt * per_core_Nt;
              uint32_t read_size = core_n_tiles * single_tile_size;

              std::vector<uint32_t> core_data_tiles;
              tt_metal::detail::ReadFromDeviceL1(device, core, out_addr,
                                                 read_size, core_data_tiles);

              auto core_data_float = unpack_bfp8_tiles_into_float_vec(
                  core_data_tiles, /*row_major_output=*/true,
                  /*is_exp_a=*/false);
                  auto core_data_untilized = untilize_compat(
                  core_data_float, per_core_Mt * 32, per_core_Nt * 32);

              uint32_t global_r_start = y * per_core_Mt * 32;
              uint32_t global_c_start = x * per_core_Nt * 32;
              uint32_t r_len = per_core_Mt * 32;
              uint32_t c_len = per_core_Nt * 32;

              for (uint32_t r = 0; r < r_len; ++r) {
                for (uint32_t c = 0; c < c_len; ++c) {
                  uint32_t global_idx =
                      (global_r_start + r) * (params.N) + (global_c_start + c);
                  if (global_idx < device_vec.size()) {
                    device_vec[global_idx] = core_data_untilized[r * c_len + c];
                  }
                }
              }
            }
          }
        }
      }

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
  } catch (const std::exception &e) {
    pass = false;
    log_error(LogTest, "{}", e.what());
  }
  return pass;
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
    ZoneScopedN("HostPipeline ComputeMM Functional Blocks");
    log_info(LogTest,
             "Starting Host-Only ComputeMM Pipeline Test (no kernel dispatch)");
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
      ZoneScopedN("HostPipeline ComputeMM Input Data Processing");
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
        ZoneScopedN("HostPipeline ComputeMM Transform Inputs");
        auto t0 = std::chrono::steady_clock::now();
        auto in0_tilized = tilize_compat(in0_vec, Mt * 32, Kt * 32);
        in0_packed = pack_fp32_vec_as_bfp8_tiles(in0_tilized,
                                                 /*row_major_input=*/true,
                                                 /*is_exp_a=*/false);

        auto in1_tilized = tilize_compat(in1_vec, Kt * 32, Nt * 32);
        in1_packed = pack_fp32_vec_as_bfp8_tiles(in1_tilized,
                                                 /*row_major_input=*/true,
                                                 /*is_exp_a=*/false);
        auto t1 = std::chrono::steady_clock::now();
        stats.transform_us +=
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                .count();
      }

      {
        ZoneScopedN("HostPipeline ComputeMM Host Enqueue");
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
        ZoneScopedN("HostPipeline ComputeMM Host FinishWait");
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
        ZoneScopedN("HostPipeline ComputeMM Host Post Processing");
        auto t0 = std::chrono::steady_clock::now();
        in0_roundtrip = unpack_bfp8_tiles_into_float_vec(
            in0_readback, /*row_major_output=*/true, /*is_exp_a=*/false);
        in0_roundtrip = untilize_compat(in0_roundtrip, Mt * 32, Kt * 32);

        in1_roundtrip = unpack_bfp8_tiles_into_float_vec(
            in1_readback, /*row_major_output=*/true, /*is_exp_a=*/false);
        in1_roundtrip = untilize_compat(in1_roundtrip, Kt * 32, Nt * 32);
        auto t1 = std::chrono::steady_clock::now();
        stats.inverse_transform_us +=
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                .count();
      }

      if (!params.bypass_check) {
        ZoneScopedN("HostPipeline ComputeMM Validation Metrics");
        float in0_pcc = get_pcc(in0_vec, in0_roundtrip);
        float in1_pcc = get_pcc(in1_vec, in1_roundtrip);
        if (i == 0) {
          log_validation_sample_pairs("HostPipeline ComputeMM IN0",
                                      in0_vec, in0_roundtrip, 12);
          log_validation_sample_pairs("HostPipeline ComputeMM IN1",
                                      in1_vec, in1_roundtrip, 12);
        }
        if (in0_pcc < 0.99f || in1_pcc < 0.99f) {
          log_error(LogTest,
                    "Host-only ComputeMM roundtrip check failed at iter {}: "
                    "in0_pcc={:.4f}, in1_pcc={:.4f}",
                    i, in0_pcc, in1_pcc);
          pass = false;
          break;
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

bool test_host_pipeline_empty_tensor(tt::tt_metal::IDevice *device,
                                     const TestParams &params) {
  bool pass = true;
  try {
  ZoneScopedN("HostPipeline Empty Functional Blocks");
  log_info(LogTest,
       "Starting Host-Only Empty Tensor Pipeline Test (no kernel dispatch)");
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
        ZoneScopedN("HostPipeline Empty Transform Inputs");
        auto t0 = std::chrono::steady_clock::now();
        auto tilized = tilize_compat(tensor_vec, Mt * 32, Nt * 32);
        packed = pack_fp32_vec_as_bfp8_tiles(tilized,
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
        ZoneScopedN("HostPipeline Empty Host Enqueue");
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
        ZoneScopedN("HostPipeline Empty Host FinishWait");
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
        ZoneScopedN("HostPipeline Empty Host Post Processing");
        auto t0 = std::chrono::steady_clock::now();
        roundtrip = unpack_bfp8_tiles_into_float_vec(
            readback, /*row_major_output=*/true, /*is_exp_a=*/false);
        roundtrip = untilize_compat(roundtrip, Mt * 32, Nt * 32);
        auto t1 = std::chrono::steady_clock::now();
        stats.inverse_transform_us +=
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                .count();
      }

      if (!params.bypass_check) {
        ZoneScopedN("HostPipeline Empty Validation Metrics");
        float pcc = get_pcc(tensor_vec, roundtrip);
        if (i == 0) {
          log_validation_sample_pairs("HostPipeline Empty", tensor_vec,
                                      roundtrip, 12);
        }
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