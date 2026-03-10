# Tenstorrent Full Characterization Benchmark

<!-- markdownlint-disable MD033 -->

This benchmark suite (`13_full_characterization`) is designed to characterize Host overheads, Data Movement, and Dispatching efficiency on Tenstorrent architectures using the `MeshDevice` and `MeshWorkload` APIs.

## Test Types

The benchmark supports the following test types, selected via the `--test` argument:

1. **Empty Kernel Launch (`--test 0`)**:
    * **Purpose**: Measures the pure overhead of dispatching and launching kernels with minimal compute/data movement.
    * **Mechanism**: Launches empty reader/writer/compute kernels.

2. **Compute MM (`--test 1`)**:
    * **Purpose**: Benchmarks the "Host Overhead" (Data Generation, Tiling, Transfer) and "Dispatch Overhead" for a standard Matrix Multiplication.
    * **Mechanism**: Runs a dense MatMul (M x N x K).
    * **Kernels**: Uses `tile_layout` kernels for reading/writing and `bmm_large_block...` for compute.
    * **Supports**: Single-Core (1x1) or Multi-Core execution (kernels are unified).

3. **Sub-Device Parallelism (`--test 2`)**:
    * **Mechanism**: Splits the grid into N "Sub-Devices" (via `--core-groups`, default 2) and dispatches independent MatMul workloads to each using a single Program with disjoint `CoreRange`s.
    * **Requirement**: Requires at least N rows of cores (`--y_size >= core_groups`).
    * **Advanced Capabilities**: While this benchmark calculates uniform divisions (e.g., 8 rows / 4 groups = 2 rows/group), Tenstorrent architectures natively support **fully uneven and non-contiguous partitions**.
        * You can define arbitrary `CoreRangeSet`s (collections of disjoint `CoreRange` rectangles) to create sub-devices of varying sizes and shapes.
        * Example: Group 1 could use a 4x4 block while Group 2 uses the remaining L-shaped set of cores. This allows modifying `test_full_charac.cpp` to support specific heterogeneous workload scenarios if needed.

4. **Host Pipeline ComputeMM (`--test 3`)**:
    * **Purpose**: Measures host-only overhead for the ComputeMM-style tensor pipeline without creating/dispatching any kernels.
    * **Mechanism**: FP32 generation → tilize → BFP8 pack → DRAM write → DRAM read → BFP8 unpack → untilize.
    * **Output**: Per-stage timing (`generate`, `transform`, `write`, `read`, `inverse_transform`, `end_to_end`) and transfer throughput.

5. **Host Pipeline Empty Tensor (`--test 4`)**:
    * **Purpose**: Measures host-only overhead using a single tensor pipeline as a lightweight baseline, with no kernel dispatch.
    * **Mechanism**: Same host transform + DRAM write/read + inverse transform path, but for one tensor only.
    * **Output**: Stage timing and throughput to isolate pure host/data-path scaling behavior.

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

* `generate`: host-side random tensor generation.
* `transform`: tilize + pack to BFP8 format.
* `write`: DRAM write time for packed tensor payloads.
* `read`: DRAM read time for packed tensor payloads.
* `inverse_transform`: unpack + untilize back to row-major FP32.
* `end_to_end`: full iteration wall time.
* Throughput summary: total bytes and effective GB/s for write and read.

Current constraints:

* Host-only tests currently use the BFP8 pipeline path (`--dtype 0`).
* `--num-iters` must be greater than 0.

### Partitioning Logic Examples

The current logic uses integer division (`rows / groups`) to assign rows. Any remainder rows are assigned to the **last group**. Here is how an **8x8 Grid** (64 cores) is partitioned with different `--core-groups`:

#### 1. Perfect Split (2 Groups)

* `--y_size 8 --core-groups 2`
* **Math**: `8 / 2 = 4` rows per group.
* **Result**:
  * **Group 0**: Rows 0-3 (8x4 grid, 32 cores)
  * **Group 1**: Rows 4-7 (8x4 grid, 32 cores)

#### 2. Uneven Split (3 Groups)

* `--y_size 8 --core-groups 3`
* **Math**: `8 / 3 = 2` rows per group (integer division). Remainder goes to the last group.
* **Result**:
  * **Group 0**: Rows 0-1 (8x2 grid, 16 cores)
  * **Group 1**: Rows 2-3 (8x2 grid, 16 cores)
  * **Group 2**: Rows 4-7 (8x4 grid, 32 cores) *(absorbs remainder)*

#### 3. Row-Level Turn (8 Groups)

* `--y_size 8 --core-groups 8`
* **Math**: `8 / 8 = 1` row per group.
* **Result**:
  * **Groups 0-7**: Each gets exactly 1 row (8x1 grid, 8 cores each).

## API Translation: New vs Legacy

This section summarizes the concrete translations needed when porting between:

* **New API path**: `test_full_charac.cpp` (`MeshDevice` / `distributed::*`)
* **Legacy API path**: `test_full_charac_old.cpp` (`IDevice` / single-device enqueue)

| Functional Area | New API (`test_full_charac.cpp`) | Legacy API (`test_full_charac_old.cpp`) | Translation / Required Change |
| :--- | :--- | :--- | :--- |
| Device handle type | `std::shared_ptr<tt::tt_metal::distributed::MeshDevice>` | `tt::tt_metal::IDevice*` | Replace mesh pointer ownership with raw `IDevice*` lifecycle (`CreateDevice` / `CloseDevice`). |
| Device creation | `MeshDevice::create_unit_mesh(device_id)` | `tt_metal::CreateDevice(device_id)` | Use legacy creator and explicit close in `main`. |
| Device close | `device->close()` | `tt_metal::CloseDevice(device)` | Convert from method-close to free-function close. |
| Command queue access | `device->mesh_command_queue()` | `device->command_queue()` | Replace mesh queue getter with legacy queue getter. |
| Program container for dispatch | `distributed::MeshWorkload` + `add_program(...)` | `tt_metal::Program` directly | Remove workload wrapper and enqueue the program directly. |
| Program enqueue | `distributed::EnqueueMeshWorkload(queue, mesh_workload, false)` | `EnqueueProgram(queue, program, false)` | One-to-one replacement in dispatch loops. |
| Host/device synchronization | `distributed::Finish(queue)` | `Finish(queue)` | Namespace/function variant differs; behavior intent is the same. |
| Program cache toggle | `device->enable_program_cache()` / `disable_and_clear_program_cache()` | same methods on `IDevice*` | No semantic change required. |
| Persistent kernel cache | `detail::EnablePersistentKernelCache()` / disable | same detail APIs | No API translation needed; keep placement consistent in `main`. |
| Host-only DRAM buffer path | `CreateBuffer`, `detail::WriteToBuffer`, `detail::ReadFromBuffer` | same calls | No translation needed for host-only tests (`--test 3`, `--test 4`). |
| Empty-kernel dispatch tracing | `MeshWorkload` + enqueue/finish split zones | `EnqueueProgram` + `Finish` split zones | Keep identical Tracy zone names so traces are directly comparable. |
| Sub-device model | Implemented via `CoreRange` splits inside mesh-based flow | **Not yet implemented** | Requires new legacy implementation; currently reported as known gap. |
| ComputeMM full kernel path | Implemented in modern file | Implemented (L1 + DRAM execution paths) | Legacy path now dispatches full ComputeMM kernels with host-side golden/reference validation. |

### Rationale & Limitations (From API Docs)

* **Do not mix models within one flow**: avoid interleaving `MeshDevice`-style dispatch and legacy single-device dispatch in the same execution path.
* **Mesh is lock-step oriented**: mesh workloads are designed to execute in coordinated fashion across mesh participants; benchmark logic should reflect that assumption.
* **Memory model differs at scale**: mesh-oriented allocation patterns are more constrained/structured than ad-hoc per-device legacy allocations; ports should avoid assuming arbitrary per-device buffer layouts.
* **Sync semantics should stay explicit**: when translating, preserve enqueue/wait boundaries (`Enqueue*` vs `Finish`) because those boundaries are what Tracy and host-overhead analysis rely on.
* **Feature parity is not automatic**: modern distributed flows can expose capabilities not yet implemented in the legacy benchmark path (for this project, `SubDevice` in old file is still a gap).

### Tracy Zone Unification Rules

When adding or porting code, preserve the same functional hierarchy in both files:

* Top-level function block (for example `ComputeMM Functional Blocks`, `EmptyKernel Functional Blocks`)
* Input/setup block
* Dispatch block with per-iteration + enqueue/wait sub-zones
* Post-processing/validation block

This ensures old/new traces remain diffable with minimal manual interpretation.

## Build and Run

Use the provided shell script `run_full_charac.sh` to compile and run the tests. This script handles kernel generation and copying based on the selected test type.

```bash
./run_full_charac.sh [path_to_executable] [arguments...]
```

### CLI Quick Reference (Current)

All currently parsed options are listed below.

<details>
<summary><strong>Show CLI options table</strong></summary>

| Group | Option | Default | Applies To | Description |
| :--- | :--- | :--- | :--- | :--- |
| Test Select | `--test <0..5>` | `5` | all | Test ID: `0` EmptyKernel, `1` ComputeMM, `2` SubDevice, `3` HostPipelineComputeMM, `4` HostPipelineEmpty (`5` = invalid sentinel). |
| Matrix | `--m <N>` | `11264` | test 1,2,3,4 | Matrix/tensor M dimension. |
| Matrix | `--n <N>` | `3072` | test 1,2,3,4 | Matrix/tensor N dimension. |
| Matrix | `--k <N>` | `768` | test 1,2,3,4 | Matrix/tensor K dimension. |
| Precision | `--dtype <0-1>` | `0` | test 1,2,3,4 | Data format selector (`0` BFP8, `1` FP16). Host-only paths currently enforce `0`. |
| Precision | `--fidel <0-1>` | `0` | test 1,2 | Math fidelity selector. |
| Layout/IO | `--dram` | off | test 1 | Enable DRAM-backed tensor path/kernels for ComputeMM. |
| Cache | `--cache` | off | all | Enable program cache + persistent kernel cache lifecycle in benchmark run. |
| Cache | `--clean-mode <0-1>` | `0` | all | Cache experiment mode; `1` invalidates cache benefits for that run. |
| Grid | `--x_size <N>` | `0` | test 0,1,2 | Core grid X (columns). |
| Grid | `--y_size <N>` | `0` | test 0,1,2 | Core grid Y (rows). |
| Grid | `--core_groups <N>` | `1` | test 0,2 | Number of core groups (sub-partitions). Must be `> 0`. |
| Runtime | `--num-iters <N>` | `15` | all | Benchmark iteration count. |
| Runtime | `--num-rt-args <N>` | `255` | test 0 | Number of runtime args used by empty-kernel setup path. |
| Runtime | `--cpu <id>` | `0xFFFFFFFF` | all | Optional CPU affinity pinning. Sentinel means no pinning. |
| Validation | `--bypass-check` | off | test 1,3,4 | Skip correctness checks (PCC/RMSE and visual validation samples). |

</details>

Notes:

* If both `--x_size` and `--y_size` are `0`, the current implementation defaults to single-core `1x1` execution.
* `--core_groups` must be `<= --y_size` when explicit grid sizes are provided.

### Example Commands

<details>
<summary><strong>Show examples (grouped)</strong></summary>

#### Empty / Baseline

```bash
./run_full_charac.sh ./build/test/test_full_charac --test 0 --num-iters 50
```

#### ComputeMM

```bash
# Single-core
./run_full_charac.sh ./build/test/test_full_charac --test 1 --x_size 1 --y_size 1 --m 512 --n 512 --k 512

# 8x8 grid
./run_full_charac.sh ./build/test/test_full_charac --test 1 --x_size 8 --y_size 8

# DRAM mode + cache enabled
./run_full_charac.sh ./build/test/test_full_charac --test 1 --dram --cache --x_size 8 --y_size 8 --m 4096 --n 4096 --k 4096
```

#### Sub-Device

```bash
./run_full_charac.sh ./build/test/test_full_charac --test 2 --x_size 8 --y_size 4 --core_groups 2
```

#### Host-only Pipelines

```bash
# ComputeMM host pipeline
./run_full_charac.sh ./build/test/test_full_charac --test 3 --num-iters 20 --m 4096 --n 4096 --k 4096 --bypass-check

# Empty tensor host pipeline
./run_full_charac.sh ./build/test/test_full_charac --test 4 --num-iters 20 --m 4096 --n 4096 --k 4096 --bypass-check
```

#### CPU pinning (optional)

```bash
./run_full_charac.sh ./build/test/test_full_charac --test 0 --num-iters 50 --cpu 3
```

</details>

## Profiling

This benchmark is instrumented with **Tracy**. To verify performance:

1. Compile with Tracy enabled (`-DTRACY_ENABLE=ON`).
2. Run the benchmark with the Tracy server (`Tracy-release`) open.
3. Inspect the complete zone inventory below (exact names as present in code).

Use the Tracy GUI to measure the duration of these zones.

### Complete Tracy Zone Inventory (Exact Names)

Inventory is grouped by test area using collapsible sections to keep this document compact.

<details>
<summary><strong>Shared Utility Zones</strong></summary>

| Tracy Zone (exact string) | Present In | Scope Contents (what is timed) |
| :--- | :--- | :--- |
| `Generate FP32 Random` | new | FP32 host random tensor generation helper body. |

</details>

<details>
<summary><strong>Test 1: ComputeMM</strong></summary>

| Tracy Zone (exact string) | Present In | Scope Contents (what is timed) |
| :--- | :--- | :--- |
| `Prepare Inputs Compute MM` | new | Full input-preparation function for ComputeMM. |
| `Prepare DRAM Inputs` | new | DRAM-specific branch inside ComputeMM input preparation. |
| `Tilize and Pack IN0 (DRAM)` | new | IN0 tilize + BFP8 pack + buffer write path setup. |
| `Tilize and Pack IN1 (DRAM)` | new | IN1 tilize + BFP8 pack + buffer write path setup. |
| `Initialize DRAM in2 zero tile` | new | Writes per-core zero tile for DRAM mode auxiliary CB input. |
| `Slicing and Tilizing IN0` | new | L1-mode per-core IN0 slicing + tilize + pack. |
| `Generating and Tilizing IN1` | new | L1-mode identity-like IN1 generation + tilize + pack. |
| `Host->Device Transfer (L1 Write)` | new | Per-core L1 writes for IN0/IN1/in2 in L1 mode. |
| `ComputeMM Functional Blocks` | new, old | Top-level ComputeMM benchmark functional block. |
| `ComputeMM Input Data Processing` | new, old | ComputeMM input/setup-oriented phase. |
| `ComputeMM Host Setup and Blocking` | new, old | Arch/tile/blocking/address derivation phase. |
| `ComputeMM Host Prepare Inputs` | new, old | Host-side tensor/data preparation before dispatch/readback. |
| `ComputeMM Host Resolve Buffer Addresses` | new, old | DRAM-vs-L1 effective address resolution step. |
| `ComputeMM Host Program Build` | new, old | Program/kernel/cb build call for ComputeMM execution path. |
| `ComputeMM Host Dispatch` | new, old | Dispatch phase wrapper for enqueue/wait operations. |
| `ComputeMM Host Dispatch Iteration` | new, old | Per-iteration dispatch scope (includes iteration `ZoneValue`). |
| `ComputeMM Host Enqueue` | new, old | Enqueue call itself (mesh enqueue or program enqueue). |
| `ComputeMM Host FinishWait` | new, old | Queue/device finish synchronization wait. |
| `ComputeMM Host Post Processing` | new, old | Post-dispatch validation/readback wrapper scope. |
| `ComputeMM Host Golden Reference` | new, old | FP32 golden matmul reference computation. |
| `ComputeMM Host Device Readback and Decode` | new, old | Device output readback + unpack/untilize/decode path. |
| `ComputeMM Host Validation Metrics` | new, old | PCC/RMSE validation metric computation/checking block. |

</details>

<details>
<summary><strong>Test 2: Sub-Device</strong></summary>

| Tracy Zone (exact string) | Present In | Scope Contents (what is timed) |
| :--- | :--- | :--- |
| `Sub-Device Parallel Dispatch` | new | Iteration-level enqueue + finish for sub-device benchmark test. |

</details>

<details>
<summary><strong>Test 3: Host Pipeline ComputeMM</strong></summary>

| Tracy Zone (exact string) | Present In | Scope Contents (what is timed) |
| :--- | :--- | :--- |
| `HostPipeline ComputeMM Functional Blocks` | new, old | Top-level host-pipeline ComputeMM benchmark block. |
| `HostPipeline ComputeMM Host Dispatch` | new, old | Iteration loop wrapper for host-only pipeline operations. |
| `HostPipeline ComputeMM Iteration` | new, old | Per-iteration wrapper (includes iteration `ZoneValue`). |
| `HostPipeline ComputeMM Prepare Inputs` | new, old | FP32 input tensor generation stage. |
| `HostPipeline ComputeMM Transform Inputs` | new, old | Tilize + BFP8 pack stage for both input tensors. |
| `HostPipeline ComputeMM Host Enqueue` | new, old | Host write stage (`WriteToBuffer`) for both tensors. |
| `HostPipeline ComputeMM Host FinishWait` | new, old | Host read stage (`ReadFromBuffer`) for both tensors. |
| `HostPipeline ComputeMM Host Post Processing` | new, old | BFP8 unpack + untilize reconstruction stage. |
| `HostPipeline ComputeMM Validation Metrics` | new, old | PCC checks for host-only ComputeMM pipeline roundtrip. |

</details>

<details>
<summary><strong>Test 4: Host Pipeline Empty Tensor</strong></summary>

| Tracy Zone (exact string) | Present In | Scope Contents (what is timed) |
| :--- | :--- | :--- |
| `HostPipeline Empty Functional Blocks` | new, old | Top-level host-pipeline empty-tensor benchmark block. |
| `HostPipeline Empty Host Dispatch` | new, old | Iteration loop wrapper for host-only pipeline operations. |
| `HostPipeline Empty Iteration` | new, old | Per-iteration wrapper (includes iteration `ZoneValue`). |
| `HostPipeline Empty Prepare Inputs` | new, old | FP32 tensor generation stage. |
| `HostPipeline Empty Transform Inputs` | new, old | Tilize + BFP8 pack stage for single tensor. |
| `HostPipeline Empty Host Enqueue` | new, old | Host write stage (`WriteToBuffer`) for tensor. |
| `HostPipeline Empty Host FinishWait` | new, old | Host read stage (`ReadFromBuffer`) for tensor. |
| `HostPipeline Empty Host Post Processing` | new, old | BFP8 unpack + untilize reconstruction stage. |
| `HostPipeline Empty Validation Metrics` | new, old | PCC checks for host-only empty-tensor pipeline roundtrip. |

</details>

<details>
<summary><strong>Test 0: Empty Kernel Launch</strong></summary>

| Tracy Zone (exact string) | Present In | Scope Contents (what is timed) |
| :--- | :--- | :--- |
| `EmptyKernel Functional Blocks` | new, old | Top-level EmptyKernel benchmark functional block. |
| `EmptyKernel Host Setup` | new, old | Program setup: CB creation, kernel creation, runtime args. |
| `EmptyKernel Host Dispatch` | new, old | Empty-kernel dispatch loop wrapper. |
| `EmptyKernel Host Dispatch Iteration` | new, old | Per-iteration empty-kernel dispatch scope. |
| `EmptyKernel Host Enqueue` | new, old | Empty-kernel enqueue call only. |
| `EmptyKernel Host FinishWait` | new, old | Empty-kernel finish/synchronization wait. |

</details>

Validation-zone policy currently enforced:

* PCC/RMSE metric calculation/checking is in dedicated validation zones (`ComputeMM Host Validation Metrics`, `HostPipeline ComputeMM Validation Metrics`, `HostPipeline Empty Validation Metrics`).
* This keeps validation math out of enqueue/wait timing zones so dispatch overhead analysis remains clean.
* Visual sanity samples are also emitted during validation (12 aligned elements), printing `ref`, `obs`, and `abs_err` to help quick manual inspection alongside PCC/RMSE.

### Dispatch Mode Requirement (Critical)

This benchmark suite is intended to run in **fast dispatch** mode.

* `TT_METAL_SLOW_DISPATCH_MODE` **must be unset** when running characterization tests.
* Both binaries explicitly reject slow dispatch at startup because it changes execution semantics of host dispatch timing.

Why this matters:

* In fast dispatch, host enqueue is asynchronous and `Finish(...)` is the synchronization boundary.
* In slow dispatch, host flow is more synchronous/direct, so measured dispatch and wait costs are not comparable to fast-dispatch data.
* Tracy zones such as `Host Enqueue` and `Host FinishWait` lose their intended interpretation under slow dispatch.

Quick check before running:

```bash
echo $TT_METAL_SLOW_DISPATCH_MODE
```

If set, clear it in your shell session:

```bash
unset TT_METAL_SLOW_DISPATCH_MODE
```

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

* **Wormhole N150**: Features a single ASIC with up to 72 usable Tensix cores. Logical grid varies by harvesting but **8x9** is the unharvested max.
* **Wormhole N300**: Features two ASICs. Each ASIC is harvested to exactly **64 cores** (typically **8x8** logical grid) to ensure consistent performance across chips.
* **Grayskull E150**: Features 120 usable cores in a 12x10 grid. E75 variant is harvested to ~88 cores.

### N150 vs N300 Impact

* **Grid Size**: If running on N150, you *might* have access to an 8x9 grid. On N300 (Device 0), you are likely limited to 8x8.
* **This Benchmark**: Runs on **Device 0** only. On an N300, this subjects the test to the thermal environment of a dual-chip card, but execution logic remains identical to N150.

### Critical Considerations

1. **Harvesting Check**: If you set `--x_size 0 --y_size 0` (default), the benchmark automatically queries `device->compute_with_storage_grid_size()` to use the maximum available unharvested grid. This is the recommended way to avoid runtime errors.
2. **Sub-Device Test Requirements**: The `SubDeviceMM` test (`--test 2`) strictly partitions the grid along the Y-axis. It **requires at least 2 unharvested rows**. If your harvested chip has `y_size < 2`, this test will fail.
3. **Data Type precision**: `BFLOAT8_B` (BFP8) is the standard for high-performance inference. `BFLOAT16` is used when higher precision is needed but has 2x memory footprint and lower math throughput.

## Validation Status (Current Debugging Campaign)

The table below tracks configurations explicitly validated or currently under validation.

| Test ID | Test Name | Mode | Scope / Shape | Status | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `--test 0` | Empty Kernel Launch | N/A | Standard dispatch path | ✅ Proven working | Stable baseline; dispatch-only path works as expected. |
| `--test 1` | ComputeMM | `L1` | Small tensor sizes | ✅ Proven working | L1 path works for small sizes. |
| `--test 1` | ComputeMM | `DRAM` | `6x6` grid | ✅ Proven working | Dispatch + execution complete successfully on Wormhole hardware. |
| `--test 1` | ComputeMM | `DRAM` | `1x1` grid | ⚠️ Not yet passing | Fails with completion-queue dispatch error (`CQ_DISPATCH_CMD_ILLEGAL`). |
| `--test 1` | ComputeMM | `DRAM` | `2x2` grid | ⚠️ Not yet passing | Same failure signature as `1x1` in current debug state. |
| `--test 3` | Host Pipeline ComputeMM | Host-only | FP32→BFP8→DRAM→BFP8→FP32 | ✅ Included in validation scope | Implemented and ready for scaling/overhead campaigns (no kernel dispatch). |
| `--test 4` | Host Pipeline Empty Tensor | Host-only | Single-tensor host pipeline | ✅ Included in validation scope | Implemented and ready for baseline host overhead studies. |
| `--test 1` | ComputeMM | `DRAM` | Large matrices (e.g. `4096x4096x4096`) | ✅ Proven working | Full MeshBuffer API migration (`MeshBuffer::create`, `EnqueueWriteMeshBuffer`, `ReadShard`, `get_device_buffer()->address()`). Dynamic L1 block solver prevents CB overflow. |
| `--test 3` | Host Pipeline ComputeMM | Host-only | MeshBuffer distributed API | ✅ Proven working | Migrated from `CreateBuffer`/`EnqueueWriteBuffer` to `MeshBuffer::create`/`EnqueueWriteMeshBuffer`/`ReadShard`. |
| `--test 4` | Host Pipeline Empty Tensor | Host-only | MeshBuffer distributed API | ✅ Proven working | Migrated from `EnqueueWriteBuffer`/`EnqueueReadBuffer` to `EnqueueWriteMeshBuffer`/`ReadShard`. |

Notes:

* This section is intentionally conservative and updated as new runs are confirmed.
* `--test 2` (SubDevice) and unlisted shape/grid combinations remain to be validated in this campaign.

## Code-Level Analysis of Test 3 and Test 4

This section analyzes the implemented host-only paths in `test_full_charac.cpp` / `test_full_charac_old.cpp` and summarizes the tensor-size math used by the code.

### Tensor Shapes and Tile Counts Used by the Code

Both tests first align dimensions to tile boundaries via `get_aligned_input_tile_num(...)`:

$$
M_t = \left\lceil \frac{M}{32} \right\rceil,\quad
N_t = \left\lceil \frac{N}{32} \right\rceil,\quad
K_t = \left\lceil \frac{K}{32} \right\rceil
$$

The code then computes tile counts:

* Test 3 (`--test 3`, ComputeMM host pipeline):
  * `in0` tiles: $M_t \cdot K_t$
  * `in1` tiles: $K_t \cdot N_t$
* Test 4 (`--test 4`, Empty host pipeline):
  * `tensor` tiles: $M_t \cdot N_t$

Packed bytes are computed exactly as in code:

$$
B = N_{\mathrm{tiles}} \cdot S
$$

where $S = \mathrm{tile\_size}(\mathrm{Bfp8\_b})$ is read from TT-Metal runtime (`tt::tile_size(...)`) and not hardcoded in the benchmark.

### Generic Formula Summary

Let:

$$
\alpha(x) = \left\lceil \frac{x}{32} \right\rceil,
\qquad
S = \mathrm{tile\_size}(\mathrm{Bfp8\_b}),
\qquad
I = \mathrm{num\_iters}
$$

Then:

* Test 3 (`MxK` + `KxN` host pipeline)

$$
B_{\mathrm{iter},3} = 2\,\big(\alpha(M)\alpha(K) + \alpha(K)\alpha(N)\big)\,S
$$

$$
B_{\mathrm{run},3} = I\,B_{\mathrm{iter},3}
$$

* Test 4 (`MxN` single-tensor host pipeline)

$$
B_{\mathrm{iter},4} = 2\,\alpha(M)\alpha(N)\,S
$$

$$
B_{\mathrm{run},4} = I\,B_{\mathrm{iter},4}
$$

The benchmark reports effective bandwidth from accumulated bytes and accumulated measured transfer time:

$$
\mathrm{BW} = \frac{B_{\mathrm{total}}}{T_{\mathrm{total}}}
$$

### Example with Current Defaults (`M=11264, N=3072, K=768`)

With 32x32 tiles:

$$
M_t=352,\;N_t=96,\;K_t=24
$$

So:

* Test 3 tiles per direction: $352\cdot24 + 24\cdot96 = 10752$
* Test 4 tiles per direction: $352\cdot96 = 33792$

If `Bfp8_b` tile size is 1088 B (DeepWiki/TT-Metal convention), then:

* Test 3 one-way bytes: $10752\cdot1088 = 11{,}698{,}176$ B
* Test 3 roundtrip bytes/iter: $23{,}396{,}352$ B
* Test 4 one-way bytes: $33792\cdot1088 = 36{,}765{,}696$ B
* Test 4 roundtrip bytes/iter: $73{,}531{,}392$ B

### Tiling and BFP8 Conversion Subprocesses

Code path used by both tests:

1. FP32 generation (`generate_fp32_random`)
2. tilize (`tilize_swizzled`) to 32x32 tile layout
3. BFP8 pack (`pack_as_bfp8_tiles`) to `Bfp8_b`
4. DRAM write/read (`WriteToBuffer` / `ReadFromBuffer`)
5. unpack (`unpack_bfp8_tiles_into_float_vec`) + untilize (`untilize_swizzled`)

DeepWiki-backed summary (repo: `tenstorrent/tt-metal`):

* Tile granularity is 32x32 elements.
* BFP8 packing stores values using block-floating representation with shared exponent groups and quantized mantissas.
* Host preprocessing adds tile-boundary padding when dimensions are not multiples of 32.

Conceptual BFP8 quantization model for a block of values $x_i$:

$$
E = \max_i \left\lfloor \log_2\left(|x_i|\right) \right\rfloor,
\qquad
m_i = \mathrm{round}\!\left(\frac{x_i}{2^E}\cdot 2^p\right)
$$

where $E$ is the shared exponent for the block and $m_i$ is the quantized mantissa/sign payload (with format-specific bit allocation in `Bfp8_b`).

Practical implication for these tests:

* `transform` time includes both layout conversion and BFP8 quantization.
* `inverse_transform` includes BFP8 decode and reverse layout conversion.
* PCC checks validate roundtrip numerical fidelity after quantize/dequantize and layout transforms.
