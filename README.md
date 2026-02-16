# Tenstorrent Full Characterization Benchmark

This benchmark suite (`13_full_characterization`) is designed to characterize Host overheads, Data Movement, and Dispatching efficiency on Tenstorrent architectures using the `MeshDevice` and `MeshWorkload` APIs.

## Test Types

The benchmark supports the following test types, selected via the `--test` argument:

1.  **Empty Kernel Launch (`--test 0`)**:
    *   **Purpose**: Measures the pure overhead of dispatching and launching kernels with minimal compute/data movement.
    *   **Mechanism**: Launches empty reader/writer/compute kernels.

2.  **Compute MM (`--test 1`)**:
    *   **Purpose**: Benchmarks the "Host Overhead" (Data Generation, Tiling, Transfer) and "Dispatch Overhead" for a standard Matrix Multiplication.
    *   **Mechanism**: Runs a dense MatMul (M x N x K).
    *   **Kernels**: Uses `tile_layout` kernels for reading/writing and `bmm_large_block...` for compute.
    *   **Supports**: Single-Core (1x1) or Multi-Core execution (kernels are unified).

3.  **Sub-Device Parallelism (`--test 2`)**:
    *   **Mechanism**: Splits the grid into N "Sub-Devices" (via `--core-groups`, default 2) and dispatches independent MatMul workloads to each using a single Program with disjoint `CoreRange`s.
    *   **Requirement**: Requires at least N rows of cores (`--y_size >= core_groups`).
    *   **Advanced Capabilities**: While this benchmark calculates uniform divisions (e.g., 8 rows / 4 groups = 2 rows/group), Tenstorrent architectures natively support **fully uneven and non-contiguous partitions**.
        *   You can define arbitrary `CoreRangeSet`s (collections of disjoint `CoreRange` rectangles) to create sub-devices of varying sizes and shapes.
        *   Example: Group 1 could use a 4x4 block while Group 2 uses the remaining L-shaped set of cores. This allows modifying `test_full_charac.cpp` to support specific heterogeneous workload scenarios if needed.

4.  **Host Pipeline ComputeMM (`--test 3`)**:
    *   **Purpose**: Measures host-only overhead for the ComputeMM-style tensor pipeline without creating/dispatching any kernels.
    *   **Mechanism**: FP32 generation → tilize → BFP8 pack → DRAM write → DRAM read → BFP8 unpack → untilize.
    *   **Output**: Per-stage timing (`generate`, `transform`, `write`, `read`, `inverse_transform`, `end_to_end`) and transfer throughput.

5.  **Host Pipeline Empty Tensor (`--test 4`)**:
    *   **Purpose**: Measures host-only overhead using a single tensor pipeline as a lightweight baseline, with no kernel dispatch.
    *   **Mechanism**: Same host transform + DRAM write/read + inverse transform path, but for one tensor only.
    *   **Output**: Stage timing and throughput to isolate pure host/data-path scaling behavior.

## Host-Only Tests Deep Dive (`--test 3` vs `--test 4`)

Both tests are **host-only data-path benchmarks**. They do not create a `Program`, do not create kernels, and do not dispatch workloads. They are intended to isolate host preprocessing and host<->device data movement costs.

### Shared Pipeline Stages

Each iteration executes:
1. Generate FP32 tensor(s) on host.
2. Transform to device-ready layout (`tilize_swizzled` + `pack_as_bfp8_tiles`).
3. Write packed tensor(s) to DRAM (`WriteToBuffer`).
4. Read packed tensor(s) back from DRAM (`ReadFromBuffer`).
5. Inverse-transform (`unpack_bfp8_tiles_into_float_vec` + `untilize_swizzled`).
6. Optionally validate roundtrip correlation (disabled with `--bypass-check`).

### What Differentiates Them

| Aspect | `--test 3` HostPipelineComputeMM | `--test 4` HostPipelineEmpty |
| :--- | :--- | :--- |
| Tensor count | 2 tensors (`in0`: MxK, `in1`: KxN) | 1 tensor (MxN) |
| Goal | Emulate full ComputeMM input pipeline overhead | Provide minimal baseline for host data path |
| Transfer volume per iteration | `bytes(in0) + bytes(in1)` for write/read | `bytes(tensor)` for write/read |
| Validation | PCC on both roundtrips (`in0`, `in1`) | PCC on single roundtrip |
| Best use | End-to-end host preprocessing scaling for matmul workloads | Compare against test 3 to estimate second-tensor overhead |

### Reported Metrics

- `generate`: host-side random tensor generation.
- `transform`: tilize + pack to BFP8 format.
- `write`: DRAM write time for packed tensor payloads.
- `read`: DRAM read time for packed tensor payloads.
- `inverse_transform`: unpack + untilize back to row-major FP32.
- `end_to_end`: full iteration wall time.
- Throughput summary: total bytes and effective GB/s for write and read.

Current constraints:
- Host-only tests currently use the BFP8 pipeline path (`--dtype 0`).
- `--num-iters` must be greater than 0.

## Validation Status (Current Debugging Campaign)

The table below tracks **only configurations explicitly tested in this repository session**.

| Test | Mode | Grid | Status | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `--test 1` (ComputeMM) | `--dram` | `6x6` | ✅ Proven working | Dispatch + execution complete successfully on Wormhole hardware. |
| `--test 1` (ComputeMM) | `--dram` | `1x1` | ⚠️ Not yet passing | Fails with completion-queue dispatch error (`CQ_DISPATCH_CMD_ILLEGAL`). |
| `--test 1` (ComputeMM) | `--dram` | `2x2` | ⚠️ Not yet passing | Same failure signature as `1x1` in current debug state. |

Notes:
- This section is intentionally conservative and only lists runs that were explicitly reported/tested.
- Other combinations (`--test 0`, `--test 2`, host-only tests, L1 mode variants, different matrix shapes) are not marked as validated here yet.

### Partitioning Logic Examples
The current logic uses integer division (`rows / groups`) to assign rows. Any remainder rows are assigned to the **last group**. Here is how an **8x8 Grid** (64 cores) is partitioned with different `--core-groups`:

**1. Perfect Split (2 Groups)**
*   `--y_size 8 --core-groups 2`
*   **Math**: `8 / 2 = 4` rows per group.
*   **Result**:
    *   **Group 0**: Rows 0-3 (8x4 grid, 32 cores)
    *   **Group 1**: Rows 4-7 (8x4 grid, 32 cores)

**2. Uneven Split (3 Groups)**
*   `--y_size 8 --core-groups 3`
*   **Math**: `8 / 3 = 2` rows per group (integer division). Remainder goes to the last group.
*   **Result**:
    *   **Group 0**: Rows 0-1 (8x2 grid, 16 cores)
    *   **Group 1**: Rows 2-3 (8x2 grid, 16 cores)
    *   **Group 2**: Rows 4-7 (8x4 grid, 32 cores) *<-- Absorbs remainder*

**3. Row-Level Turn (8 Groups)**
*   `--y_size 8 --core-groups 8`
*   **Math**: `8 / 8 = 1` row per group.
*   **Result**:
    *   **Groups 0-7**: Each gets exactly 1 row (8x1 grid, 8 cores each).

## Build and Run

Use the provided shell script `run_full_charac.sh` to compile and run the tests. This script handles kernel generation and copying based on the selected test type.

```bash
./run_full_charac.sh [path_to_executable] [arguments...]
```

### Common Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--test <ID>` | `5` | Test Type ID (0=Empty, 1=ComputeMM, 2=SubDevice, 3=HostPipelineComputeMM, 4=HostPipelineEmpty). |
| `--x_size <N>` | `0` (Max) | Number of columns in the core grid. |
| `--y_size <N>` | `0` (Max) | Number of rows in the core grid. |
| `--num-iters <N>` | `15` | Number of iterations to run the dispatch loop. |
| `--core-groups <N>`| `2` | Number of sub-devices/splits for Test 2. |
| `--clean-mode <0/1>`| `0` | If 1, cleans kernel cache before running. |

### Matrix Arguments (for ComputeMM and SubDeviceMM)

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--m <N>` | `11264` | M dimension of the matrix. |
| `--n <N>` | `3072` | N dimension of the matrix. |
| `--k <N>` | `768` | K dimension of the matrix. |
| `--dtype <0/1>` | `0` | Data type (0=BFP8, 1=FP16). |
| `--fidel <0/1>` | `0` | Math fidelity (0=LoFi, 1=HiFi). |

## Examples

**1. Run Empty Kernel Launch on full grid:**
```bash
./run_full_charac.sh ./build/test/test_full_charac --test 0 --num-iters 50
```

**2. Run Compute MM on a 1x1 grid (Single Core Benchmarking):**
```bash
./run_full_charac.sh ./build/test/test_full_charac --test 1 --x_size 1 --y_size 1 --m 512 --n 512 --k 512
```

**3. Run Compute MM on a 8x8 grid:**
```bash
./run_full_charac.sh ./build/test/test_full_charac --test 1 --x_size 8 --y_size 8
```

**4. Run Sub-Device Parallelism Test (Requires >= 2 rows):**
```bash
./run_full_charac.sh ./build/test/test_full_charac --test 2 --x_size 8 --y_size 4
```

**5. Run Host-Only ComputeMM Pipeline (no kernels):**
```bash
./run_full_charac.sh ./build/test/test_full_charac --test 3 --num-iters 20 --m 4096 --n 4096 --k 4096 --bypass-check
```

**6. Run Host-Only Empty Tensor Pipeline (no kernels):**
```bash
./run_full_charac.sh ./build/test/test_full_charac --test 4 --num-iters 20 --m 4096 --n 4096 --k 4096 --bypass-check
```

## Profiling

This benchmark is instrumented with **Tracy**. To verify performance:
1.  Compile with Tracy enabled (`-DTRACY_ENABLE=ON`).
2.  Run the benchmark with the Tracy server (`Tracy-release`) open.
3.  Look for zones like:
    *   `Host->Device Transfer (L1 Write)`
    *   `Dispatch Overhead`
    *   `Sub-Device Parallel Dispatch`
    *   `Prepare Inputs Compute MM`

Use the Tracy GUI to measure the duration of these zones.

## Architecture Specifications & Constraints

It is critical to set `--x_size` and `--y_size` within the bounds of your specific device's available compute grid. The table below lists the **Logical Compute Grids** (the grid of Tensix cores exposed to the program).

| Feature | Grayskull (E150) | Wormhole (N150) | Wormhole (N300*) | Blackhole (BH) |
| :--- | :--- | :--- | :--- | :--- |
| **Physical Tile Grid** | 10 x 12 | 10 x 12 | 10 x 12 (x2 chips) | ~17 x 12 |
| **Max Logical Compute Grid** | **12 x 10** (120 Cores) | **8 x 9** (72 Cores)** | **8 x 8** (64 Cores/Chip) | **14 x 10** (140 Cores) |
| **L1 Memory / Core** | 1024 KB | 1484 KB | 1484 KB | > 1484 KB |
| **Compute Data Types** | `BFLOAT16`, `BFLOAT8_B`, `BFLOAT4_B` (No FP32) | `BFLOAT16`, `BFLOAT8_B`, `BFLOAT4_B`, `FLOAT32` | `BFLOAT16`, `BFLOAT8_B`, `BFLOAT4_B`, `FLOAT32` | All + `TF32` |

*\*N300 is a dual-chip card. Specs listed are PER CHIP.*
*\*\*N150 is a single-chip card. While the unharvested grid is 8x9 (72 cores), manufacturing harvesting often reduces this (e.g., to 8x8 or 8x7).*

### Device Variants & Grid Sizes
*   **Wormhole N150**: Features a single ASIC with up to 72 usable Tensix cores. Logical grid varies by harvesting but **8x9** is the unharvested max.
*   **Wormhole N300**: Features two ASICs. Each ASIC is harvested to exactly **64 cores** (typically **8x8** logical grid) to ensure consistent performance across chips.
*   **Grayskull E150**: Features 120 usable cores in a 12x10 grid. E75 variant is harvested to ~88 cores.

### N150 vs N300 Impact
*   **Grid Size**: If running on N150, you *might* have access to an 8x9 grid. On N300 (Device 0), you are likely limited to 8x8.
*   **This Benchmark**: Runs on **Device 0** only. On an N300, this subjects the test to the thermal environment of a dual-chip card, but execution logic remains identical to N150.

### Critical Considerations
1.  **Harvesting Check**: If you set `--x_size 0 --y_size 0` (default), the benchmark automatically queries `device->compute_with_storage_grid_size()` to use the maximum available unharvested grid. This is the recommended way to avoid runtime errors.
2.  **Sub-Device Test Requirements**: The `SubDeviceMM` test (`--test 2`) strictly partitions the grid along the Y-axis. It **requires at least 2 unharvested rows**. If your harvested chip has `y_size < 2`, this test will fail.
3.  **Data Type precision**: `BFLOAT8_B` (BFP8) is the standard for high-performance inference. `BFLOAT16` is used when higher precision is needed but has 2x memory footprint and lower math throughput.

