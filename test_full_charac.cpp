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

// Logging
#include <tt-logger/tt-logger.hpp>

// Tracy
#include <tracy/Tracy.hpp>
#include <tt-metalium/tilize_utils.hpp>

using namespace tt;
using namespace tt::tt_metal;

////////////////////////////////////////////////////////////////////////////////////////
// Default values for benchmark parameters
////////////////////////////////////////////////////////////////////////////////////////

constexpr uint32_t DEFAULT_ITERATIONS = 10000;
// constexpr uint32_t DEFAULT_WARMUP_ITERATIONS = 100;
// constexpr uint32_t MIN_KERNEL_SIZE_BYTES = 32;  // overhead
// constexpr uint32_t DEFAULT_KERNEL_SIZE_K = 1;
// constexpr uint32_t MAX_CBS = 32;
constexpr uint32_t MAX_ARGS = 255;
constexpr uint32_t L1_SAFETY_MARGIN_BYTES = 64 * 1024;

enum class TestType : uint32_t {
  EmptyKernelLaunch = 0,
  ComputeMM = 1,
  SubDeviceMM = 2,
  InvalidTest = 3
};

// Definition of test parameters structure
struct TestParams {
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t dtype; // 0: BFP8, 1: FP16
  uint32_t fidel; // 0: low, 1: high
  bool use_dram;  // Enable DRAM input buffers
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
  std::shared_ptr<tt::tt_metal::distributed::MeshDevice> device;
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
                                                                "--test", 3);
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
    uint32_t in2_cb_size = single_tile_size;
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

    if (base_addr + total_l1_needed + L1_SAFETY_MARGIN_BYTES <= l1_size) {
      in0_block_w = choice;
      break;
    }
  }
  return in0_block_w;
}

// TTNN-aligned L1 fitting for DRAM mode.
// Finds the largest (out_block_h, out_block_w, in0_block_w) such that the
// circular buffers fit within available L1.
//
// Algorithm (matching TTNN's get_multi_dim_per_core_factor):
//   1. Enumerate all divisor pairs of per_core_M and per_core_N
//   2. Sort by product (largest first), then by squareness
//   3. For each (obh, obw), try in0_block_w from Kt down to 1
//   4. Return the first combination where CBs fit L1
//
// CB budget: in0_CB + in1_CB + in2_CB + out_CB + interm_CB <= avail_L1
//   in0_CB = out_block_h * in0_block_w * 2 * tile_size  (double-buffered)
//   in1_CB = out_block_w * in0_block_w * 2 * tile_size  (double-buffered)
//   in2_CB = tile_size                                    (zeros tile)
//   out_CB = out_block_h * out_block_w * tile_size        (output block)
//   interm = out_block_h * out_block_w * tile_size        (FP32 partials)
//
// Returns {out_block_h, out_block_w, in0_block_w}
std::tuple<uint32_t, uint32_t, uint32_t>
get_multi_dim_per_core_factor(uint32_t per_core_M, uint32_t per_core_N,
                              uint32_t Kt, uint32_t single_tile_size,
                              uint32_t l1_size, uint32_t l1_unreserved_base) {
  uint32_t raw_avail_l1 = l1_size - l1_unreserved_base;
  uint32_t avail_l1 =
      (raw_avail_l1 > L1_SAFETY_MARGIN_BYTES)
          ? (raw_avail_l1 - L1_SAFETY_MARGIN_BYTES)
          : raw_avail_l1;

  // Helper: check if a given (obh, obw, bw) fits in L1
  auto fits_l1 = [&](uint32_t obh, uint32_t obw, uint32_t bw) -> bool {
    uint32_t in0_cb = obh * bw * 2 * single_tile_size; // double-buffered
    uint32_t in1_cb = obw * bw * 2 * single_tile_size; // double-buffered
    uint32_t in2_cb = single_tile_size;                // zeros
    uint32_t out_cb = obh * obw * single_tile_size;    // output
    uint32_t interm = obh * obw * single_tile_size;    // FP32 partials
    return (in0_cb + in1_cb + in2_cb + out_cb + interm) <= avail_l1;
  };

  // Try full dimensions first
  if (fits_l1(per_core_M, per_core_N, Kt)) {
    return {per_core_M, per_core_N, Kt};
  }

  // Enumerate divisors
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

  // Build (m, n) pairs sorted by product descending, then squareness
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
  // Sort: largest product first, then most square
  std::sort(pairs.begin(), pairs.end(), [](const Pair &a, const Pair &b) {
    if (a.product != b.product)
      return a.product > b.product;
    return a.ratio < b.ratio;
  });

  // Try each (m, n) pair with decreasing in0_block_w
  for (auto &p : pairs) {
    // Try in0_block_w from largest possible down to 1
    // Must be a divisor of Kt (or at least work with integer K-blocks)
    for (uint32_t bw = Kt; bw >= 1; bw--) {
      if (Kt % bw != 0)
        continue;
      if (fits_l1(p.m, p.n, bw)) {
        return {p.m, p.n, bw};
      }
    }
  }

  // Fallback: 1x1 with bw=1 should always fit
  return {1, 1, 1};
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

  // Device-side DRAM interleaved buffers.
  // These are nullptr when use_dram=false (L1 mode).
  // When use_dram=true, each buffer holds tilized+packed data distributed
  // across all DRAM banks via InterleavedBufferConfig{BufferType::DRAM}.
  std::shared_ptr<tt_metal::Buffer> in0_buffer; // IN0 (A matrix) in DRAM
  std::shared_ptr<tt_metal::Buffer> in1_buffer; // IN1 (B matrix) in DRAM
  std::shared_ptr<tt_metal::Buffer> out_buffer; // Output (C matrix) in DRAM
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
    uint32_t start_core_y = 0);

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

    // 1. Get Common Params
    auto arch = device->arch();
    uint32_t l1_size = get_l1_size(arch);
    uint32_t l1_unreserved_base =
        device->allocator()->get_base_allocator_addr(HalMemType::L1);
    auto [math_fidelity, fp32_dest_acc_en] = get_compute_params(arch);
    tt::DataFormat data_format = (params.dtype == 0)
                                     ? tt::DataFormat::Bfp8_b
                                     : tt::DataFormat::Float16_b;
    uint32_t single_tile_size = tt::tile_size(data_format);

    // 2. Prepare Program with Loop over Splits
    tt_metal::Program program;
    uint32_t M_split = params.M / num_sub_devices;

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

      uint32_t in0_block_w =
          get_in0_block_w(per_core_Mt, per_core_Nt, Kt, single_tile_size,
                          l1_size, l1_unreserved_base);
      if (in0_block_w == 0)
        throw std::runtime_error("Insufficient L1 for SubDevice split");

      auto [out_subblock_h, out_subblock_w] =
          get_out_subblock_params(per_core_Mt, per_core_Nt);

      auto [in0_cb_addr, in1_cb_addr, in2_cb_addr, out_cb_addr, interm_cb_addr,
            in0_addr, in1_addr, out_addr] =
          get_all_buffers_addresses(per_core_Mt, per_core_Nt, in0_block_w,
                                    single_tile_size, l1_unreserved_base);

      // Create Kernels/Buffers (Appends to program using strict CoreRange)
      // L1 mode: out_block_h/w == per_core_Mt/Nt, num_blocks = 1
      create_program_compute_mm(
          device, data_format, math_fidelity, fp32_dest_acc_en,
          single_tile_size, split_grid_size, Mt, Nt, Kt, in0_block_w,
          out_subblock_h, out_subblock_w, per_core_Mt, per_core_Nt, per_core_Mt,
          per_core_Nt, 1, 1, in0_cb_addr, in1_cb_addr, in2_cb_addr, out_cb_addr,
          interm_cb_addr, in0_addr, in1_addr, out_addr,
          /*use_dram=*/false, program, start_y);

      // Prepare Inputs for this split
      // Prepare Inputs for this split
      auto inputs_split = prepare_inputs_compute_mm(
          device, split_grid_size, Mt, Nt, Kt, per_core_Mt, per_core_Nt,
          in0_block_w, single_tile_size, in0_addr, in1_addr, in2_cb_addr,
          /*use_dram=*/false, start_y);
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

// 1-to-1 match with 1_compute_mm/test_compute_mm.cpp
// MODIFIED: FP32 ACCUM TO TRUE -> poor preceision
std::tuple<MathFidelity, bool> get_compute_params(tt::ARCH arch) {
  MathFidelity math_fidelity = MathFidelity::HiFi4;
  bool fp32_dest_acc_en = true;
  if (arch == tt::ARCH::WORMHOLE_B0 or arch == tt::ARCH::BLACKHOLE) {
    math_fidelity = MathFidelity::HiFi2;
    fp32_dest_acc_en = true;
  } else if (arch == tt::ARCH::GRAYSKULL) {
    math_fidelity = MathFidelity::HiFi4;
    fp32_dest_acc_en = true;
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
  uint32_t in2_cb_size = single_tile_size;
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

// Reference Matrix Multiplication (row-major)
// C = A * B. A is (M x K), B is (K x N), C is (M x N)
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
BenchmarkInputs prepare_inputs_compute_mm(
    tt_metal::distributed::MeshDevice *device, CoreCoord core_range,
    uint32_t Mt, uint32_t Nt, uint32_t Kt, uint32_t per_core_Mt,
    uint32_t per_core_Nt, uint32_t in0_block_w, uint32_t single_tile_size,
    uint32_t in0_addr, uint32_t in1_addr, uint32_t in2_cb_addr, bool use_dram,
    uint32_t start_core_y) {

  ZoneScopedN("Prepare Inputs Compute MM");
  BenchmarkInputs inputs;

  // Generate random FP32 matrices. These are kept in the BenchmarkInputs struct
  // for later host-side golden-reference validation (matmul_reference).
  // TILE_HW = 32*32 = 1024 elements per tile.
  // Scale factor for ML-realistic initialization: range [-5, 5] (User Request)
  float scale = 5.0f;
  inputs.in0_vec = generate_fp32_random(Mt * Kt * constants::TILE_HW, scale);
  inputs.in1_vec = generate_fp32_random(Nt * Kt * constants::TILE_HW, scale);

  // Zeros buffer for the "in2" circular buffer (CB index 2).
  // This is written to every core's L1 regardless of L1/DRAM mode.
  // The compute kernel reads from CB2 for bias/padding initialization;
  // we fill it with zeros since we don't use bias in this benchmark.
  std::vector<uint32_t> in2(single_tile_size / sizeof(uint32_t), 0);
  auto *target_device = device->get_devices()[0];

  if (use_dram) {
    ZoneScopedN("Prepare DRAM Inputs");

    // ---- DRAM Step 1: Tilize and pack IN0 (full matrix A) ----
    // tilize_swizzled() converts from row-major to the hardware tile layout
    // (32x32 element groups in Z-order). pack_as_bfp8_tiles() then quantizes
    // the FP32 data into BFP8_b format (block floating point, 8-bit mantissa).
    // We tilize the ENTIRE matrix at once (Mt*32 × Kt*32), unlike L1 mode
    // which slices per-core.
    {
      ZoneScopedN("Tilize and Pack IN0 (DRAM)");
      auto in0_tilized = tilize_swizzled(inputs.in0_vec, Mt * 32, Kt * 32);
      auto in0_packed =
          pack_as_bfp8_tiles(tt::stl::make_const_span(in0_tilized),
                             /*row_major_input=*/true, /*is_exp_a=*/false);

      // Create a DRAM interleaved buffer to hold Mt*Kt tiles.
      // InterleavedBufferConfig tells the allocator to distribute tiles
      // across DRAM banks in round-robin order. Each "page" = one tile.
      // The buffer's base address (buffer->address()) will be passed to the
      // kernel as a runtime argument for InterleavedAddrGenFast<true>.
      uint32_t in0_num_tiles = Mt * Kt;
      uint32_t in0_size_bytes = in0_num_tiles * single_tile_size;
      inputs.in0_buffer =
          tt_metal::CreateBuffer(tt_metal::InterleavedBufferConfig{
              .device = target_device,
              .size = in0_size_bytes,
              .page_size = single_tile_size, // one tile per page
              .buffer_type = tt_metal::BufferType::DRAM});

      // WriteToBuffer writes the packed data to the interleaved buffer.
      // Internally, this writes each page (tile) to the correct DRAM bank
      // based on the round-robin interleaving scheme.
      tt_metal::detail::WriteToBuffer(inputs.in0_buffer, in0_packed);

      // Update inputs.in0_vec with effective BFP8 values for validation
      auto in0_unpacked = unpack_bfp8_tiles_into_float_vec(
          in0_packed, /*row_major_output=*/true, /*is_exp_a=*/false);
      inputs.in0_vec = untilize_swizzled(in0_unpacked, Mt * 32, Kt * 32);
    }

    // ---- DRAM Step 2: Tilize and pack IN1 (full matrix B) ----
    // Same flow as IN0: tilize → pack BFP8 → create DRAM buffer → write.
    // IN1 has dimensions Kt*32 (rows) × Nt*32 (cols), totaling Kt*Nt tiles.
    {
      ZoneScopedN("Tilize and Pack IN1 (DRAM)");
      auto in1_tilized = tilize_swizzled(inputs.in1_vec, Kt * 32, Nt * 32);
      auto in1_packed =
          pack_as_bfp8_tiles(tt::stl::make_const_span(in1_tilized),
                             /*row_major_input=*/true, /*is_exp_a=*/false);

      // Update inputs.in1_vec with effective BFP8 values for validation
      auto in1_unpacked = unpack_bfp8_tiles_into_float_vec(
          in1_packed, /*row_major_output=*/true, /*is_exp_a=*/false);
      inputs.in1_vec = untilize_swizzled(in1_unpacked, Kt * 32, Nt * 32);

      uint32_t in1_num_tiles = Kt * Nt;
      uint32_t in1_size_bytes = in1_num_tiles * single_tile_size;
      inputs.in1_buffer =
          tt_metal::CreateBuffer(tt_metal::InterleavedBufferConfig{
              .device = target_device,
              .size = in1_size_bytes,
              .page_size = single_tile_size, // one tile per page
              .buffer_type = tt_metal::BufferType::DRAM});
      tt_metal::detail::WriteToBuffer(inputs.in1_buffer, in1_packed);
    }

    // ---- DRAM Step 3: Create empty output buffer ----
    // The output buffer holds the result C = A*B. Total tiles = Mt * Nt.
    // The kernel will write to this buffer using InterleavedAddrGenFast<true>
    // and noc_async_write_tile().
    {
      uint32_t out_num_tiles = Mt * Nt;
      uint32_t out_size_bytes = out_num_tiles * single_tile_size;
      inputs.out_buffer =
          tt_metal::CreateBuffer(tt_metal::InterleavedBufferConfig{
              .device = target_device,
              .size = out_size_bytes,
              .page_size = single_tile_size,
              .buffer_type = tt_metal::BufferType::DRAM});
    }

    // ---- DRAM Step 4: No host-side L1 write for in2 ----
    // DRAM reader/writer kernels reserve/use cb_id_in2 directly via
    // get_write_ptr(cb_id_in2). Writing to host-computed in2_cb_addr here is
    // redundant and may be unsafe on versions where CB addresses are
    // allocator-assigned rather than host-fixed.

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
        auto in0_block_tilized =
            tilize_swizzled(in0_block_slice, num_r * 32, in0_block_w * 32);
        in0 = pack_as_bfp8_tiles(tt::stl::make_const_span(in0_block_tilized),
                                 /*row_major_input=*/true, /*is_exp_a=*/false);
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

          auto in1_block_tilized =
              tilize_swizzled(in1_block_slice, in0_block_w * 32, num_c * 32);
          in1 =
              pack_as_bfp8_tiles(tt::stl::make_const_span(in1_block_tilized),
                                 /*row_major_input=*/true, /*is_exp_a=*/false);
        }

        // Transfer tile blocks to this core's L1 SRAM.
        // WriteToDeviceL1 writes directly to a specific address in the target
        // core's L1 memory. The kernel reads from these addresses using local
        // NOC self-read (noc_async_read with phy_core coordinates).
        {
          ZoneScopedN("Host->Device Transfer (L1 Write)");
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
  //   CB2 (c_2):  Bias/padding tile      — single tile (zeros)
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
      tt_metal::CircularBufferConfig(single_tile_size,
                                     {{tt::CBIndex::c_2, cb_data_format}})
          .set_page_size(tt::CBIndex::c_2, single_tile_size);
  tt_metal::CreateCircularBuffer(program, all_cores, cb_src2);

  // Separate allocations for Output (cb_16) and Intermediate (cb_24)
  // to avoid aliasing and deadlock.
  std::map<uint8_t, tt::DataFormat> cb_out_config_map = {
      {(uint8_t)tt::CBIndex::c_16, cb_data_format}};
  tt_metal::CircularBufferConfig cb_out_config =
      tt_metal::CircularBufferConfig(out_CB_size, cb_out_config_map)
          .set_page_size(tt::CBIndex::c_16, single_tile_size);
  tt_metal::CreateCircularBuffer(program, CoreRangeSet({all_cores}),
                                 cb_out_config);

  std::map<uint8_t, tt::DataFormat> cb_interm_config_map = {
      {(uint8_t)tt::CBIndex::c_24, cb_data_format}};
  tt_metal::CircularBufferConfig cb_interm_config =
      tt_metal::CircularBufferConfig(out_CB_size, cb_interm_config_map)
          .set_page_size(tt::CBIndex::c_24, single_tile_size);
  tt_metal::CreateCircularBuffer(program, CoreRangeSet({all_cores}),
                                 cb_interm_config);

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
    reader_kernel_path = "tests/tt_metal/tt_metal/perf_microbenchmark/"
                         "13_full_charac/kernels/"
                         "in0_reader_bmm_tile_layout_dram.cpp";
    writer_kernel_path = "tests/tt_metal/tt_metal/perf_microbenchmark/"
                         "13_full_charac/kernels/"
                         "in1_reader_writer_bmm_tile_layout_dram.cpp";
    // No IN1_IS_IDENTITY for DRAM — we read real matrix data from DRAM
  } else {
    // L1 kernels: read from local L1 SRAM.
    // IN1_IS_IDENTITY define tells the writer kernel to reuse the same
    // IN1 block for all K-block iterations (identity matrix workaround).
    // NOTE: The shell script copies these from kernels_common/ into kernels/.
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

  // Compute kernel: Tensix FPU math engine (same for both L1/DRAM modes).
  // The compute kernel operates entirely on L1 circular buffers,
  // so it doesn't need to know whether data came from L1 or DRAM.
  // NOTE: Shell script always copies this from kernels_common/ into kernels/.
  tt_metal::CreateKernel(
      program,
      "tests/tt_metal/tt_metal/perf_microbenchmark/13_full_charac/kernels/"
      "bmm_large_block_zm_fused_bias_activation.cpp",
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
        // === DRAM Reader Args (15 args) ===
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
            last_outer_num_nonzero_subblocks_h, // 24: last bh nonzero sbh
            last_outer_subblock_h,              // 25: last bh tail sbh height
            last_outer_padded_block_tiles_h_skip, // 26: last bh padded rows
            last_outer_num_nonzero_subblocks_w, // 27: last bw nonzero sbw
            last_outer_subblock_w,              // 28: last bw tail sbw width
            last_outer_padded_subblock_tiles_addr_skip, // 29: tail sbw skip
            last_outer_padded_block_tiles_w_skip,       // 30: last bw pad skip
            num_blocks_h,              // 31: num_blocks_h_dim
            num_blocks_w,              // 32: num_blocks_w_dim
            out_block_w,               // 33: in1_w_dim_stride
            out_block_h * Nt,          // 34: out_h_dim_stride
            out_block_w,               // 35: out_w_dim_stride
        };

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
                       out_subblock_w * out_subblock_h,
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

///////////////////////////////////////////////////////////////////////////////////////////
/// Compute MM Test
///////////////////////////////////////////////////////////////////////////////////////////
bool test_compute_mm(tt::tt_metal::distributed::MeshDevice *device,
                     const TestParams &params) {
  bool pass = true;
  try {
    log_info(LogTest, "Starting Compute MM Test");
    log_info(LogTest, "M={}, N={}, K={}", params.M, params.N, params.K);

    // 1. Get L1 size and arch params
    auto arch = device->arch();
    uint32_t l1_size = get_l1_size(arch);
    uint32_t l1_unreserved_base =
        device->allocator()->get_base_allocator_addr(HalMemType::L1);
    auto [math_fidelity, fp32_dest_acc_en] = get_compute_params(arch);

    // 2. Calculate blocking parameters
    auto [Mt, Nt, Kt] =
        get_aligned_input_tile_num(params.M, params.N, params.K);
    uint32_t num_cores_x = params.core_x;
    uint32_t num_cores_y = params.core_y;
    CoreCoord core_range(num_cores_x, num_cores_y);

    uint32_t per_core_Mt = ((Mt - 1) / num_cores_y) + 1;
    uint32_t per_core_Nt = ((Nt - 1) / num_cores_x) + 1;

    tt::DataFormat data_format = (params.dtype == 0)
                                     ? tt::DataFormat::Bfp8_b
                                     : tt::DataFormat::Float16_b;
    uint32_t single_tile_size = tt::tile_size(data_format);

    // ---- TTNN-aligned L1 fitting for DRAM mode ----
    // Use get_multi_dim_per_core_factor to find optimal (out_block_h,
    // out_block_w, in0_block_w). The kernel-internal bh × bw loops handle the
    // iteration over output blocks, so the host only creates a single program.
    uint32_t out_block_h, out_block_w, in0_block_w;
    uint32_t num_blocks_h = 1, num_blocks_w = 1;

    if (params.use_dram) {
      // DRAM mode: CBs sized by out_block_h × out_block_w (may be < per_core)
      auto [obh, obw, bw] = get_multi_dim_per_core_factor(
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
               per_core_Mt, per_core_Nt, out_block_h, out_block_w, num_blocks_h,
               num_blocks_w, in0_block_w, L1_SAFETY_MARGIN_BYTES / 1024);
    } else {
      // L1 mode: full per-core dims fit (no multi-block needed)
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

    // Subblock params based on out_block_h/w
    auto [out_subblock_h, out_subblock_w] =
        get_out_subblock_params(out_block_h, out_block_w);

    // 3. Buffer Addresses (use out_block dims for CB sizing)
    auto [in0_cb_addr, in1_cb_addr, in2_cb_addr, out_cb_addr, interm_cb_addr,
          in0_addr, in1_addr, out_addr] =
        get_all_buffers_addresses(out_block_h, out_block_w, in0_block_w,
                                  single_tile_size, l1_unreserved_base,
                                  params.use_dram);

    // 4. Prepare Inputs — always use FULL per_core dims for DRAM buffer sizing
    auto inputs = prepare_inputs_compute_mm(
        device, core_range, Mt, Nt, Kt, per_core_Mt, per_core_Nt, in0_block_w,
        single_tile_size, in0_addr, in1_addr, in2_cb_addr, params.use_dram);

    // 5. Resolve addresses: DRAM buffers override L1 addresses
    uint32_t effective_in0_addr = in0_addr;
    uint32_t effective_in1_addr = in1_addr;
    uint32_t effective_out_addr = out_addr;
    if (params.use_dram) {
      effective_in0_addr = inputs.in0_buffer->address();
      effective_in1_addr = inputs.in1_buffer->address();
      effective_out_addr = inputs.out_buffer->address();
      log_info(LogTest, "DRAM mode: in0=0x{:x}, in1=0x{:x}, out=0x{:x}",
               effective_in0_addr, effective_in1_addr, effective_out_addr);
    }

    // 6. Create Program and Run (single program — kernel handles multi-block)
    log_info(LogTest, "Num tests {}", params.num_iters);

    tt_metal::Program program;
    create_program_compute_mm(
        device, data_format, math_fidelity, fp32_dest_acc_en, single_tile_size,
        core_range, Mt, Nt, Kt, in0_block_w, out_subblock_h, out_subblock_w,
        per_core_Mt, per_core_Nt, out_block_h, out_block_w, num_blocks_h,
        num_blocks_w, in0_cb_addr, in1_cb_addr, in2_cb_addr, out_cb_addr,
        interm_cb_addr, effective_in0_addr, effective_in1_addr,
        effective_out_addr, params.use_dram, program);

    auto mesh_workload = tt_metal::distributed::MeshWorkload();
    mesh_workload.add_program(
        tt::tt_metal::distributed::MeshCoordinateRange{{0, 0}, {0, 0}},
        std::move(program));

    for (uint32_t i = 0; i < params.num_iters; ++i) {
      ZoneScopedN("Dispatch Overhead");
      ZoneValue(i);
      tt_metal::distributed::EnqueueMeshWorkload(device->mesh_command_queue(),
                                                 mesh_workload, false);
      tt_metal::distributed::Finish(device->mesh_command_queue());
    }

    if (!params.bypass_check) {
      log_info(LogTest, "Validation Started...");

      // Compute Golden Reference
      log_info(LogTest, "Computing Golden Reference (FP32)...");
      auto golden_vec = matmul_reference(inputs.in0_vec, inputs.in1_vec,
                                         params.M, params.N, params.K);

      // Read Back Results
      log_info(LogTest, "Reading Device Results...");
      std::vector<float> device_vec(params.M * params.N, 0.0f);
      auto *target_device = device->get_devices()[0];

      if (params.use_dram) {
        // === DRAM Readback ===
        // Read entire output buffer from DRAM
        std::vector<uint32_t> out_data;
        tt_metal::detail::ReadFromBuffer(inputs.out_buffer, out_data);

        // Unpack and untilize the full output
        auto out_float = unpack_bfp8_tiles_into_float_vec(
            out_data, /*row_major_output=*/true, /*is_exp_a=*/false);
        // out_float is in tilized layout (Mt*32 rows x Nt*32 cols)
        device_vec = untilize_swizzled(out_float, Mt * 32, Nt * 32);
        // Trim to actual M x N if needed
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
        // === L1 Readback (per-core stitching) ===
        for (int y = 0; y < (int)num_cores_y; y++) {
          for (int x = 0; x < (int)num_cores_x; x++) {
            CoreCoord core = {(std::size_t)x, (std::size_t)y};
            uint32_t core_n_tiles = per_core_Mt * per_core_Nt;
            uint32_t read_size = core_n_tiles * single_tile_size;

            std::vector<uint32_t> core_data_tiles;
            tt_metal::detail::ReadFromDeviceL1(target_device, core, out_addr,
                                               read_size, core_data_tiles);

            // Unpack and Untilize
            auto core_data_float = unpack_bfp8_tiles_into_float_vec(
                core_data_tiles, /*row_major_output=*/true,
                /*is_exp_a=*/false);
            auto core_data_untilized = untilize_swizzled(
                core_data_float, per_core_Mt * 32, per_core_Nt * 32);

            // Copy into global device_vec
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

      // Comparison
      // Comparison
      float pcc = get_pcc(golden_vec, device_vec);
      float rmse = get_rmse(golden_vec, device_vec);
      float relative_rmse = get_relative_rmse(golden_vec, device_vec);

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

///////////////////////////////////////////////////////////////////////////////////////////
/// Empty Kernel Launch Test
///////////////////////////////////////////////////////////////////////////////////////////
bool test_empty_kernel_launch(tt::tt_metal::distributed::MeshDevice *device,
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
    auto mesh_workload = tt_metal::distributed::MeshWorkload();
    mesh_workload.add_program(
        tt::tt_metal::distributed::MeshCoordinateRange{{0, 0}, {0, 0}},
        std::move(program));

    // Explicitly compile the program to measure compile overhead WITH TRACY
    // THIS IS NOT NEEDED
    /*
    auto t_compile_begin = std::chrono::steady_clock::now();
    for (auto& [range, prog] : mesh_workload.get_programs()) {
        tt_metal::detail::CompileProgram(device->get_devices()[0], prog);
    }
    auto t_compile_end = std::chrono::steady_clock::now();
    auto compile_time =
    std::chrono::duration_cast<std::chrono::microseconds>(t_compile_end -
    t_compile_begin).count(); log_info(LogTest, "Time elapsed for
    compilation: {}us", compile_time);
    */
    // Now we should have a cache hit
    log_info(LogTest, "Num tests {}", params.num_iters);
    for (uint32_t i = 0; i < params.num_iters; ++i) {
      // auto t_begin = std::chrono::steady_clock::now();
      ZoneScopedN("Empty Kernel Launch Execution");
      ZoneValue(i);
      tt_metal::distributed::EnqueueMeshWorkload(device->mesh_command_queue(),
                                                 mesh_workload, false);
      tt_metal::distributed::Finish(device->mesh_command_queue());

      // auto t_end = std::chrono::steady_clock::now();
      // elapsed_us.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t_end
      // - t_begin).count()); log_info(LogTest, "Time elapsed for executing
      // empty kernels: {}us", elapsed_us[i]);
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

  // Create mesh device
  int device_id = 0;
  DeviceParams device_params;

  device_params.device =
      tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);
  device_params.grid_coord =
      device_params.device->compute_with_storage_grid_size();

  // Definition of max cores in each dimension for the architecture being
  // tested
  uint32_t max_x = device_params.grid_coord.x;
  uint32_t max_y = device_params.grid_coord.y;

  // Parse input arguments
  std::vector<std::string> input_args(argv, argv + argc);
  TestParams params = parse_input_arguments(input_args, device_params);

  if (params.cpu_id != 0xFFFFFFFF) {
    pin_to_cpu(params.cpu_id);
  }

  //// Print test summary
  log_info(LogTest, "Full Characterization Benchmarking Test");
  log_info(LogTest, "=======================================");
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
    pass = test_empty_kernel_launch(device_params.device.get(), params);
    break;
  case TestType::ComputeMM:
    pass = test_compute_mm(device_params.device.get(), params);
    break;
  case TestType::SubDeviceMM:
    pass = test_sub_device_manager_mm(device_params.device.get(), params);
    break;
  default:
    log_error(tt::LogTest, "Invalid test type selected: {}",
              static_cast<uint32_t>(params.test));
    return -1;
  }

  // We finalize the device
  device_params.device->close();

  return 0;
}
