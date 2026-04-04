# Tenstorrent Full Characterization Benchmark

Compact benchmark suite for host overhead, data movement, and dispatch characterization on Tenstorrent hardware.

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Environment Export Presets](#environment-export-presets)
- [Test Catalog](#test-catalog)
- [CLI Reference](#cli-reference)
- [Command Playbook](#command-playbook)
- [Dispatch Accuracy Study (Matrix32)](#dispatch-accuracy-study-matrix32)
- [RW Buffer Sweep](#rw-buffer-sweep)
- [Test 6 Trace/Replay Engineering Decisions](#test-6-tracereplay-engineering-decisions)
- [Validation and Precision Model](#validation-and-precision-model)
- [Execution Matrix (Tried)](#execution-matrix-tried)
- [Test 2 DRAM Support (Final Scope)](#test-2-dram-support-final-scope)
- [Profiling with Tracy](#profiling-with-tracy)
- [Test 1 Tracy Zone Mapping](#test-1-tracy-zone-mapping)
- [Architecture and Grid Notes](#architecture-and-grid-notes)
- [Troubleshooting](#troubleshooting)
- [Project Layout](#project-layout)
- [Appendix: Reference Notes](#appendix-reference-notes)

## Overview
This repository benchmark target is [test_full_charac.cpp](test_full_charac.cpp).

It provides:
- Empty-kernel dispatch baseline.
- End-to-end ComputeMM benchmark (L1 and DRAM paths).
- Sub-device split execution benchmark.
- Host-only pipelines to isolate transform and DRAM transfer costs.

Design goals:
- Keep dispatch timing and transfer timing separable.
- Keep CLI backward compatible while enabling explicit dtype/export controls.
- Support both CPU-side and TTNN device-side transform paths.

## Quick Start
### 1) Build and run
```bash
./run_full_charac.sh ./build/test/test_full_charac --test 0 --num-iters 20
```

### 1.5) Standalone CMake build
To run a standalone build via CMake, use CMakeLists in the project root and the default output path is usually `./build/test_full_charac`.

### 2) Dispatch mode requirement
This suite is intended for fast dispatch.

```bash
echo $TT_METAL_SLOW_DISPATCH_MODE
unset TT_METAL_SLOW_DISPATCH_MODE
```

### 3) Default execution behavior
If `--x_size 0 --y_size 0`, current code defaults to `1x1` execution.

## Environment Export Presets
These presets are based on runtime option behavior and the tests exercised in this repo (`0,1,2,5,6`, plus host pipelines `3,4`).

### 1) Mandatory baseline (out-of-tree runs)
Use this for all benchmark commands unless your shell/session already sets these.

```bash
export TT_METAL_HOME=/scratch/javier/tt-metal
export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-$TT_METAL_HOME}"
export TT_METAL_CACHE="${TT_METAL_CACHE:-$HOME/.cache/tt-metal-cache}"
unset TT_METAL_SLOW_DISPATCH_MODE
```

### 2) Accurate measurement preset (low-noise baseline)
Use this when collecting latency/throughput numbers and comparing test modes.

```bash
unset TT_METAL_WATCHER
unset TT_METAL_DPRINT_CORES
unset TT_METAL_DEVICE_PROFILER
unset TT_METAL_TRACE_PROFILER
unset TT_METAL_DISABLE_DMA_OPS
unset TT_METAL_RECORD_NOC_TRANSFER_DATA

export TT_METAL_GTEST_NUM_HW_CQS=1
export TT_METAL_NUMA_BASED_AFFINITY=1
unset TT_METAL_SLOW_DISPATCH_MODE
```

### 3) Tracy and device-profiler preset
Use this for device timeline analysis. A Tracy-enabled tt-metal build is required when enabling `TT_METAL_DEVICE_PROFILER=1`.

```bash
export TT_METAL_DEVICE_PROFILER=1
export TT_METAL_DEVICE_PROFILER_DISPATCH=1
export TT_METAL_PROFILER_SYNC=1
export TT_METAL_TRACE_PROFILER=1
export TT_METAL_PROFILER_TRACE_TRACKING=1
export TT_METAL_PROFILER_CPP_POST_PROCESS=1
export TT_METAL_TRACY_MID_RUN_PUSH=1
```

### 4) DMA and transfer-debug preset
Use this when isolating suspected DMA/data-movement issues. This mode is intentionally intrusive and can change behavior/performance.

```bash
export TT_METAL_DISABLE_DMA_OPS=1
export TT_METAL_VALIDATE_PROGRAM_BINARIES=1
export TT_METAL_RECORD_NOC_TRANSFER_DATA=1
export TT_METAL_WATCHER=500ms
export TT_METAL_WATCHER_ENABLE_NOC_SANITIZE_LINKED_TRANSACTION=1
export TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1
```

### 5) DPRINT preset (firmware-side logs)
Use this for core-local debug logs. Do not combine with device profiler in the same run.

```bash
unset TT_METAL_DEVICE_PROFILER
unset TT_METAL_TRACE_PROFILER

export TT_METAL_DPRINT_CORES=all
export TT_METAL_DPRINT_CHIPS=all
export TT_METAL_DPRINT_RISCVS=BR+NCRISC+TRISC0
export TT_METAL_DPRINT_FILE=/tmp/tt_dprint.log
export TT_METAL_DPRINT_PREPEND_DEVICE_CORE_RISC=1
```

### 6) Fast reset helper
Use this to return to clean benchmark conditions.

```bash
unset TT_METAL_DEVICE_PROFILER TT_METAL_DEVICE_PROFILER_DISPATCH TT_METAL_PROFILER_SYNC
unset TT_METAL_TRACE_PROFILER TT_METAL_PROFILER_TRACE_TRACKING TT_METAL_PROFILER_CPP_POST_PROCESS
unset TT_METAL_TRACY_MID_RUN_PUSH TT_METAL_WATCHER TT_METAL_DISABLE_DMA_OPS TT_METAL_RECORD_NOC_TRANSFER_DATA
unset TT_METAL_DPRINT_CORES TT_METAL_DPRINT_CHIPS TT_METAL_DPRINT_RISCVS TT_METAL_DPRINT_FILE
unset TT_METAL_LOG_KERNELS_COMPILE_COMMANDS TT_METAL_VALIDATE_PROGRAM_BINARIES
```

## Test Catalog
| Test ID | Name | Purpose | Dispatches Kernels |
| :-- | :-- | :-- | :-- |
| `0` | Empty Kernel Launch | Baseline dispatch overhead | Yes |
| `1` | ComputeMM | Full matmul path with optional DRAM and TTNN data transforms | Yes |
| `2` | SubDeviceMM | Split grid into core groups and dispatch independent partitions | Yes |
| `3` | HostPipelineComputeMM | Host-only two-tensor transform + DRAM write/read loop | No |
| `4` | HostPipelineEmpty | Host-only single-tensor baseline pipeline | No |
| `5` | ComputeMMAsyncBatch | ComputeMM async batch enqueue with single finish for fair baseline | Yes |
| `6` | ComputeMMTraceReplay | TTNN trace capture + replay (capture once, loop replay) | Yes |

## CLI Reference
### Core selection and runtime
| Option | Default | Notes |
| :-- | :-- | :-- |
| `--test <0..6>` | `7` | `0..4` standard tests; `5` = ComputeMMAsyncBatch, `6` = ComputeMMTraceReplay (`7` internal invalid sentinel). |
| `--num-iters <N>` | `15` | Iteration count for all tests. |
| `--x_size <N>` | `0` | Grid X. With `0,0`, code defaults to `1x1`. |
| `--y_size <N>` | `0` | Grid Y. |
| `--core_groups <N>` | `1` | Must be `> 0` and `<= y_size`. |
| `--num-rt-args <N>` | `255` | Empty-kernel runtime argument count. |
| `--cpu <id>` | `0xFFFFFFFF` | Optional CPU affinity base id. |
| `--cpu-range <N>` | `4` | CPU affinity range width. |
| `--cache` | off | Enables program and persistent kernel cache lifecycle. |
| `--clean-mode <0/1>` | `0` | Cache experiment switch. |
| `--bypass-check` | off | Skips numerical validation when enabled. |

### Shape and precision
| Option | Default | Notes |
| :-- | :-- | :-- |
| `--m <N>` | `11264` | M dimension. |
| `--n <N>` | `3072` | N dimension. |
| `--k <N>` | `768` | K dimension. |
| `--dtype <0..2>` | `0` | Legacy selector: `0=bfp8`, `1=bf16`, `2=fp32 request`. |
| `--input-dtype <bfp8|bf16|fp32|inherit>` | `inherit` | Explicit input CLI control. |
| `--output-dtype <native|bf16|fp32>` | `native` | Export/readback dtype behavior for test `1`. |
| `--fidel <0/1>` | `0` | Math fidelity selector. |

### Data path controls
| Option | Default | Notes |
| :-- | :-- | :-- |
| `--dram` | off | Enables DRAM-backed ComputeMM path. |
| `--pack-tile <cpu|device|inherit>` | `inherit` | If `inherit`, follows `--unpack-tile`. |
| `--unpack-tile <cpu|device>` | `cpu` | Device-side readback path when set to `device`. |

## Command Playbook
### Baselines
```bash
# Empty kernel baseline
./run_full_charac.sh ./build/test/test_full_charac --test 0 --num-iters 50

# ComputeMM single core
./run_full_charac.sh ./build/test/test_full_charac --test 1 --x_size 1 --y_size 1 --m 512 --n 512 --k 512
```

### ComputeMM DRAM stress
```bash
./run_full_charac.sh ./build/test/test_full_charac \
  --test 1 --dram --x_size 8 --y_size 7 \
  --m 4096 --n 4096 --k 4096 --num-iters 10
```
### ComputeMM async batch baseline (new test 5)
```bash
./run_full_charac.sh ./build/test/test_full_charac \
  --test 5 --num-iters 100 \
  --x_size 1 --y_size 1 --m 512 --n 512 --k 512 --bypass-check
```

### ComputeMM trace capture/replay (new test 6)
```bash
./run_full_charac.sh ./build/test/test_full_charac \
  --test 6 --num-iters 100 --num-rt-args 1 \
  --x_size 1 --y_size 1 --m 512 --n 512 --k 512
```


### Pack/Unpack path matrix (test 1 + DRAM)
```bash
# CPU pack, CPU unpack
./run_full_charac.sh ./build/test/test_full_charac --test 1 --dram --pack-tile cpu --unpack-tile cpu

# Device pack, CPU unpack
./run_full_charac.sh ./build/test/test_full_charac --test 1 --dram --pack-tile device --unpack-tile cpu

# CPU pack, Device unpack
./run_full_charac.sh ./build/test/test_full_charac --test 1 --dram --pack-tile cpu --unpack-tile device

# Device pack, Device unpack
./run_full_charac.sh ./build/test/test_full_charac --test 1 --dram --pack-tile device --unpack-tile device
```

### Explicit input/export dtype control
```bash
# BFP8 compute input, FP32 export readback
./run_full_charac.sh ./build/test/test_full_charac \
  --test 1 --dram --input-dtype bfp8 --output-dtype fp32 \
  --pack-tile device --unpack-tile device

# BF16 compute input, BF16 export readback
./run_full_charac.sh ./build/test/test_full_charac \
  --test 1 --dram --input-dtype bf16 --output-dtype bf16 \
  --pack-tile cpu --unpack-tile device
```

### Sub-device split
```bash
./run_full_charac.sh ./build/test/test_full_charac \
  --test 2 --x_size 8 --y_size 8 --core_groups 2 --num-iters 20
```

### Host-only pipelines
```bash
# Two-tensor host pipeline (matmul-like)
./run_full_charac.sh ./build/test/test_full_charac \
  --test 3 --m 4096 --n 4096 --k 4096 --num-iters 20

# One-tensor host baseline
./run_full_charac.sh ./build/test/test_full_charac \
  --test 4 --m 4096 --n 4096 --k 4096 --num-iters 20
```

## Dispatch Accuracy Study (Matrix32)
This section captures the exact commands used to run a precision-focused dispatch study, including multicore (`1x1` vs `7x7`) and cache policy (`warm` vs `clean-case`) with parser-safe TSV/CSV outputs.

### Script defaults (run_pgm_dispatch_host_suite.sh)
- `--mode phases`
- `--warmup 2000`
- `--iters 3000`
- `--repeats 1`
- `--phases A,B,C,D`
- `--cache-mode warm`
- `--binary ./build/test_pgm_dispatch`
- `--reset` disabled by default
- Default output directory if `--out-dir` is not set:
  - `logs/pgm_dispatch_suite_<name_config_timestamp>`
  - Config fields include mode, phases (for `phases` mode), warmup, iters, repeats, and cache mode.

### Reset option
- Use `--reset` to run `tt-smi -r 0` before each case repetition.
- `--reset-before-each` is supported as an alias.

### 1) Clean only raw logs (keep CSV/TSV analysis data)
```bash
find logs -type f -name '*.log' -delete
```

### 2) Warm-cache accurate run
```bash
./run_pgm_dispatch_host_suite.sh \
  --mode matrix32 \
  --warmup 200 \
  --iters 1000 \
  --repeats 3 \
  --cache-mode warm \
  --taskset-cpus 2-8 \
  --out-dir logs/pgm_dispatch_matrix32_accuracy_warm_v2 \
  --log-dir logs/pgm_dispatch_logs_matrix32_accuracy
```

### 3) Clean-cache-per-case accurate run
```bash
./run_pgm_dispatch_host_suite.sh \
  --mode matrix32 \
  --warmup 200 \
  --iters 1000 \
  --repeats 3 \
  --cache-mode clean-case \
  --taskset-cpus 2-8 \
  --out-dir logs/pgm_dispatch_matrix32_accuracy_clean_v2 \
  --log-dir logs/pgm_dispatch_logs_matrix32_accuracy
```

### 4) Result files for Python/Excel
- `summary.tsv` / `summary.csv`: one row per configuration.
- `measurements.tsv` / `measurements.csv`: one row per repetition.
- `matrix32_pivot.csv`: compact plotting matrix (`grid`, `runtime_args`, `tr0_f0`, `tr0_f1`, `tr1_f0`, `tr1_f1`).

### 5) Notes on stability and interpretation
- `--iters` controls internal averaging in each invocation.
- `--repeats` controls run-to-run averaging and stddev.
- `--taskset-cpus` reduces host scheduling noise.
- `--cache-mode clean-case` can expose cold-start/cache-sensitive behavior, but it may also increase variance in low-latency `finish-only` paths.

## RW Buffer Sweep
Use [run_rw_buffer_sweep.sh](run_rw_buffer_sweep.sh) to benchmark host-to-device and device-to-host bandwidth across transfer/page-size sweeps with parser-friendly TSV/CSV outputs.

### Script defaults (run_rw_buffer_sweep.sh)
- `--binary ./build/test_rw_buffer`
- `--device 0`
- `--num-tests 20`
- `--repeats 3`
- `--buffer-types 0`
- `--transfer-sizes 67108864,268435456,536870912,1073741824`
- `--page-sizes 2048,8192,32768`
- `--mode full`
- `--reset` disabled by default
- Default output directory if `--out-dir` is not set:
  - `logs/rw_buffer_sweep_<name_config_timestamp>`
  - Config fields include mode, buffer types, transfer sizes, page sizes, num-tests, and repeats.

### Reset option
- Use `--reset` to run `tt-smi -r 0` before each command repetition.

### Reproducible size-only sweep (DRAM, single core, forced page 2048)
This is the exact reproducibility recipe used for the latest size-focused campaign.

```bash
./run_rw_buffer_sweep.sh \
  --mode full \
  --device 0 \
  --buffer-types 0 \
  --transfer-sizes 8388608,67108864,268435456,536870912,1073741824 \
  --page-sizes 2048 \
  --num-tests 30 \
  --repeats 3 \
  --taskset-cpus 2 \
  --out-dir logs/rw_buffer_size_only_page2048_v1 \
  --log-dir logs/rw_buffer_logs_size_only_page2048_v1
```

### Iteration accounting for averaging
- Cases: `5` transfer sizes (`8MB, 64MB, 256MB, 512MB, 1GB`).
- Internal iterations per case: `num-tests * repeats = 30 * 3 = 90`.
- Total internal iterations: `cases * num-tests * repeats = 5 * 30 * 3 = 450`.
- In `--mode full`, each iteration runs one write and one read.
- Total transfer operations: `2 * 450 = 900`.

### Alternate heavy-workload sweep (multi-page exploration)
```bash
./run_rw_buffer_sweep.sh \
  --device 0 \
  --buffer-types 0 \
  --transfer-sizes 67108864,268435456,536870912,1073741824 \
  --page-sizes 2048,8192,32768 \
  --num-tests 20 \
  --repeats 3 \
  --taskset-cpus 2-8 \
  --out-dir logs/rw_buffer_sweep_heavy \
  --log-dir logs/rw_buffer_logs_heavy
```

### Tunable parameters and what they represent
- `--buffer-types`: `0=DRAM`, `1=L1`.
  - For realistic large-ML host IO pressure, prefer DRAM (`0`).
  - Use L1 (`1`) for on-chip residency sensitivity checks.
- `--transfer-sizes` (bytes): workload batch pressure.
  - Larger values increase sustained bandwidth sensitivity.
  - Suggested heavy range: `64MB` to `1GB`.
- `--page-sizes` (bytes): transaction granularity.
  - Smaller pages stress dispatch/descriptor overhead.
  - Larger pages emphasize sustained transfer efficiency.
- `--num-tests`: in-binary averaging window.
  - Increase to reduce measurement jitter.
- `--repeats`: external repetition for confidence intervals.
- `--mode full|write-only|read-only`:
  - `full` for balanced throughput behavior.
  - `write-only` isolates H2D path.
  - `read-only` isolates D2H path.
- `--taskset-cpus`, `--numactl-cpubind`, `--numactl-membind`: host placement controls for reproducible runs.

### Generated files
- `summary.csv`: averaged H2D/D2H and best read/write with stddev.
- `measurements.csv`: raw per-repeat measurements for plotting error bars.
- Logs are overwritten in the selected log directory each run; summary outputs remain in the selected output directory.

## Test 6 Trace/Replay Engineering Decisions
This section documents the exact decisions used to make Test 6 (`ComputeMMTraceReplay`) reliable in this repository.

- `MeshDevice` trace-region rule: trace capture/replay requires a non-zero trace region when opening the device. Upstream/default behavior is effectively `0` bytes unless explicitly set.
- Chosen trace-region size for this repo: `256 KB` (`256 * 1024`) when `--test 6` is selected; otherwise `0`.
- Chosen `num_command_queues`: `1` for this benchmark to keep ordering deterministic and avoid cross-CQ variance during capture/replay timing.
- CQ consistency rule: warmup enqueue, capture enqueue, end-capture, replay, and finish all use the same trace CQ (`cq_id = 0`).
- Script support: [run_full_charac.sh](run_full_charac.sh) routes both `--test 5` and `--test 6` through the same ComputeMM kernel-generation branch as tests `1` and `2`, then executes the benchmark command.

## Validation and Precision Model
### Validation flow
- Golden reference is computed in FP32 for relevant tests.
- DRAM + validation mode reconstructs effective packed inputs before comparison.
- Validation can be disabled with `--bypass-check`.

### Effective compute dtype mapping
Current benchmark compute path supports BFP8 and BF16 formats.

- `--input-dtype bfp8` -> effective compute format `BFP8_B`
- `--input-dtype bf16` -> effective compute format `BF16`
- `--input-dtype fp32` -> currently mapped to effective compute format `BF16`

### Output export/readback behavior (test 1)
- `--output-dtype native`: read back in native compute path.
- `--output-dtype bf16`: BF16-oriented export path.
- `--output-dtype fp32`: TTNN typecast to FLOAT32 then untilize/readback.

## Execution Matrix (Tried)
This table tracks practical configurations that have been exercised in the current campaign.

| Scope | Configuration | Status | Notes |
| :-- | :-- | :-- | :-- |
| Test 1 DRAM | `--pack-tile cpu --unpack-tile cpu` | Verified | Stable baseline DRAM path. |
| Test 1 DRAM | `--pack-tile device --unpack-tile cpu` | Verified | Device pack path validated. |
| Test 1 DRAM | `--pack-tile cpu --unpack-tile device` | Verified | Readback handle fix validated. |
| Test 1 DRAM | `--pack-tile device --unpack-tile device` | Verified | End-to-end TTNN transform path validated. |

| Additional Campaign Checks | Configuration | Status | Notes |
| :-- | :-- | :-- | :-- |
| Test 1 DRAM large shape | `--m 4096 --n 4096 --k 4096` | Verified | Used as representative stress case. |
| Test 3 host-only | Host pipeline ComputeMM | Verified | Host transform and DRAM transfer flow used for overhead isolation. |
| Test 4 host-only | Host pipeline Empty | Verified | Single-tensor host baseline flow available. |

## Test 2 DRAM Support (Final Scope)
The Test 2 DRAM adoption is intentionally isolated from other tests and uses a pragmatic scope to reduce integration risk.

### Supported now
- `--test 2 --dram` is supported for execution/dispatch.
- DRAM addressing is resolved per split from MeshBuffer device addresses.
- Per-split input/output buffer lifetimes are retained for the full dispatch loop.
- Dynamic DRAM blocking is used in Test 2 when DRAM is enabled.

### Compromises and constraints
- For Test 2 + DRAM, `M` must be divisible by `core_groups`.
- For Test 2 + DRAM, device pack is currently forced to CPU pack (risk-reduction compromise).
- Test 2 does not perform output readback/export; `--unpack-tile` and `--output-dtype` are ignored in this test.

These constraints are local to Test 2 and do not modify behavior of Test 0/1/3/4.

### Recommended commands
```bash
# DRAM smoke, single split
./run_full_charac.sh ./build/test/test_full_charac \
  --test 2 --dram --x_size 4 --y_size 4 --core_groups 1 \
  --m 1024 --n 1024 --k 1024 --num-iters 20

# DRAM multi-split (M divisible by core_groups)
./run_full_charac.sh ./build/test/test_full_charac \
  --test 2 --dram --x_size 8 --y_size 8 --core_groups 2 \
  --m 4096 --n 4096 --k 4096 --num-iters 20

# L1 regression check (unchanged behavior expected)
./run_full_charac.sh ./build/test/test_full_charac \
  --test 2 --x_size 8 --y_size 8 --core_groups 2 \
  --m 4096 --n 4096 --k 4096 --num-iters 20
```

## Profiling with Tracy
This project is instrumented with Tracy zones for:
- Host setup and blocking.
- Input preparation and transform phases.
- Dispatch enqueue vs finish wait boundaries.
- Readback and validation phases.

Recommended workflow:
1. Build with Tracy enabled.
2. Launch Tracy server.
3. Run benchmark command from this README.
4. Compare `Host Enqueue` and `Host FinishWait` regions across modes.

## Test 1 Tracy Zone Mapping
This mapping is for side-by-side trace analysis between [test_full_charac.cpp](test_full_charac.cpp) (new API) and [test_full_charac_old.cpp](test_full_charac_old.cpp) (legacy port).

Rule used:
- Exact same behavior and intent keeps the same zone name.
- Equivalent legacy-only behavior is labeled with ` (leg)` in the legacy trace.

| New API zone | Legacy zone | Mapping |
| :-- | :-- | :-- |
| `ComputeMM Functional Blocks` | `ComputeMM Functional Blocks` | Exact |
| `ComputeMM Input Data Processing` | `ComputeMM Input Data Processing` | Exact |
| `ComputeMM Host Setup and Blocking` | `ComputeMM Host Setup and Blocking` | Exact |
| `ComputeMM Host Prepare Inputs` | `ComputeMM Host Prepare Inputs` | Exact |
| `Prepare Inputs Compute MM` | `Prepare Inputs Compute MM (leg)` | Equivalent (legacy naming) |
| `Prepare Inputs On-Device (ttnn)` | `Prepare Inputs On-Device (ttnn) (leg)` | Equivalent (legacy naming) |
| `ttnn::IN0_Prepare` | `ttnn::IN0_Prepare` | Exact |
| `ttnn::IN0_ToDevice` | `ttnn::IN0_ToDevice` | Exact |
| `ttnn::IN0_TilizePack` | `ttnn::IN0_TilizePack` | Exact |
| `ttnn::IN0_CaptureBufferAndRetainTensor` | `ttnn::IN0_CaptureBufferAndRetainTensor` | Exact |
| `ttnn::IN1_Prepare` | `ttnn::IN1_Prepare` | Exact |
| `ttnn::IN1_ToDevice` | `ttnn::IN1_ToDevice` | Exact |
| `ttnn::IN1_TilizePack` | `ttnn::IN1_TilizePack` | Exact |
| `ttnn::IN1_CaptureBufferAndRetainTensor` | `ttnn::IN1_CaptureBufferAndRetainTensor` | Exact |
| `ttnn::Sync` | `ttnn::Sync` | Exact |
| `ttnn::Output_CreateTensor (Unpack Device Bootstrap)` | `ttnn::Output_CreateTensor (Unpack Device Bootstrap) (leg)` | Equivalent (legacy naming) |
| `ComputeMM Host Resolve Buffer Addresses` | `ComputeMM Host Resolve Buffer Addresses` | Exact |
| `ComputeMM Host Program Build` | `ComputeMM Host Program Build` | Exact |
| `ComputeMM Host Dispatch` | `ComputeMM Host Dispatch` | Exact |
| `ComputeMM Host Dispatch Iteration` | `ComputeMM Host Dispatch Iteration` | Exact |
| `ComputeMM Host Enqueue` | `ComputeMM Host Enqueue` | Exact |
| `ComputeMM Host FinishWait` | `ComputeMM Host FinishWait` | Exact |
| `ComputeMM Host Post Processing` | `ComputeMM Host Post Processing` | Exact |
| `ComputeMM Host Validation: Reconstruct Effective Inputs` | `ComputeMM Host Validation: Reconstruct Effective Inputs (leg)` | Equivalent (legacy naming) |
| `ComputeMM Host Validation: Decode Effective IN0 (Device Pack)` | `ComputeMM Host Validation: Decode Effective IN0 (Device Pack) (leg)` | Equivalent (legacy naming) |
| `ComputeMM Host Validation: Read Packed IN0 (Device Pack)` | `ComputeMM Host Validation: Decode Effective IN0 Readback (leg)` | Equivalent (legacy naming) |
| `ComputeMM Host Validation: Decode Effective IN1 (Device Pack)` | `ComputeMM Host Validation: Decode Effective IN1 (Device Pack) (leg)` | Equivalent (legacy naming) |
| `ComputeMM Host Validation: Read Packed IN1 (Device Pack)` | `ComputeMM Host Validation: Decode Effective IN1 Readback (leg)` | Equivalent (legacy naming) |
| `ComputeMM Host Golden Reference` | `ComputeMM Host Golden Reference` | Exact |
| `ComputeMM Host Device Readback` | `ComputeMM Host Device Readback` | Exact |
| `ComputeMM Host Device Unpack (ttnn)` | `ComputeMM Host Device Unpack (ttnn) (leg)` | Equivalent (legacy naming) |
| `ttnn::untilize` | `ttnn::untilize` | Exact |
| `ttnn::untilize_Finish` | `ttnn::untilize_Finish` | Exact |
| `ttnn::to_vector<float> (Readback)` | `ttnn::to_vector<float> (Readback)` | Exact |
| `tt-metal::Output_CreateTensor DRAM Buffer (Manual)` | `tt-metal::Output_CreateTensor DRAM Buffer (Manual) (leg)` | Equivalent (legacy naming) |
| `ComputeMM Host ReadShard DRAM Output` | `ComputeMM Host ReadShard DRAM Output (leg)` | Equivalent (legacy naming) |
| `ComputeMM Host Decode DRAM` | `ComputeMM Host Decode DRAM (leg)` | Equivalent (legacy naming) |
| `ComputeMM Host Trim DRAM Decode to MxN` | `ComputeMM Host Trim Device Readback to MxN (leg)` | Equivalent (legacy naming) |
| `ComputeMM Host Trim Device Readback to MxN` | `ComputeMM Host Trim Device Readback to MxN (leg)` | Equivalent (legacy naming) |
| `ComputeMM Host ReadFromDeviceL1 All Cores` | `ComputeMM Host ReadFromDeviceL1 All Cores (leg)` | Equivalent (legacy naming) |
| `ComputeMM Host Decode L1` | `ComputeMM Host Decode L1 (leg)` | Equivalent (legacy naming) |
| `ComputeMM Host Validation Metrics` | `ComputeMM Host Validation Metrics` | Exact |

Notes:
- Legacy deterministic profiling mode for Test 1 DRAM supports two complete pipelines only: `cpu/cpu` or `device/device`; mixed mode is rejected.
- Device-pack effective-input reconstruction in legacy now runs in post-processing validation (phase-aligned with new API) and is tracked with `(leg)`-suffixed zones.
- The table is intentionally scoped to Test 1 ComputeMM flow.

## Architecture and Grid Notes
- Always keep grid sizes within the device logical compute grid.
- Sub-device mode (`--test 2`) requires `y_size >= core_groups`.
- Harvested devices can expose fewer usable rows/cols than max silicon layout.

## Troubleshooting
### Invalid test id
Use `--test` in the `0..6` range.

### `core_y < core_groups`
Increase `--y_size` or reduce `--core_groups`.

### Unexpected precision mismatch
Check:
- `--input-dtype`
- `--output-dtype`
- `--pack-tile` and `--unpack-tile`
- `--bypass-check` state

### Slow-dispatch contamination
Unset `TT_METAL_SLOW_DISPATCH_MODE` before running benchmarks.

## Project Layout
- Main benchmark executable: [test_full_charac.cpp](test_full_charac.cpp)
- Legacy comparison path: [test_full_charac_old.cpp](test_full_charac_old.cpp)
- Main compute kernel: [kernels_common/bmm_large_block_zm_fused_bias_activation.cpp](kernels_common/bmm_large_block_zm_fused_bias_activation.cpp)
- DRAM tile readers/writers: [kernels_common/in0_reader_bmm_tile_layout_dram.cpp](kernels_common/in0_reader_bmm_tile_layout_dram.cpp), [kernels_common/in1_reader_writer_bmm_tile_layout_dram.cpp](kernels_common/in1_reader_writer_bmm_tile_layout_dram.cpp)
- Runner script: [run_full_charac.sh](run_full_charac.sh)

## Appendix: Reference Notes

<details>
<summary><strong>Hardware Reference Table (quick lookup)</strong></summary>

| Feature | Grayskull (E150) | Wormhole (N150) | Wormhole (N300, per chip) | Blackhole (BH) |
| :-- | :-- | :-- | :-- | :-- |
| Physical tile grid | 10 x 12 | 10 x 12 | 10 x 12 | ~17 x 12 |
| Max logical compute grid | 12 x 10 | 8 x 9 (unharvested max) | 8 x 8 typical | 14 x 10 |
| L1 memory / core | 1024 KB | 1484 KB | 1484 KB | >1484 KB |
| Common compute formats | BF16, BFP8_B, BFP4_B | BF16, BFP8_B, BFP4_B, FP32 | BF16, BFP8_B, BFP4_B, FP32 | BF16/BFPx/FP32/TF32 |

Notes:
- N300 is dual-chip; this benchmark path typically targets one device context at a time.
- Harvesting may reduce usable grid rows/columns versus unharvested maxima.

</details>

<details>
<summary><strong>BFLOAT8 (BFP8_B) practical notes</strong></summary>

BFP8_B is block-floating-point based quantization optimized for throughput and bandwidth efficiency.

Practical interpretation in this benchmark:
- Host or device transform path converts FP32 tensors into tile layout and packs to BFP8_B when selected.
- Validation compares against effective transformed inputs to avoid false mismatches due to quantization effects.
- BFP8_B generally improves transfer/computation efficiency, while BF16 may improve numerical fidelity at higher byte cost.

Conceptual block-floating model:

$$
E = \max_i \left\lfloor \log_2\left(|x_i|\right) \right\rfloor, \quad
m_i = \mathrm{round}\!\left(\frac{x_i}{2^E}\cdot 2^p\right)
$$

where $E$ is the shared exponent for a block and $m_i$ are quantized mantissas.

</details>

---
If you need, this README can be extended with a dedicated "experiment templates" section (for latency sweeps, grid sweeps, and precision sweeps) while preserving this compact structure.
