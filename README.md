# Tenstorrent Full Characterization Benchmark

Compact benchmark suite for host overhead, data movement, and dispatch characterization on Tenstorrent hardware.

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Test Catalog](#test-catalog)
- [CLI Reference](#cli-reference)
- [Command Playbook](#command-playbook)
- [Validation and Precision Model](#validation-and-precision-model)
- [Execution Matrix (Tried)](#execution-matrix-tried)
- [Test 2 DRAM Support (Final Scope)](#test-2-dram-support-final-scope)
- [Profiling with Tracy](#profiling-with-tracy)
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

### 2) Dispatch mode requirement
This suite is intended for fast dispatch.

```bash
echo $TT_METAL_SLOW_DISPATCH_MODE
unset TT_METAL_SLOW_DISPATCH_MODE
```

### 3) Default execution behavior
If `--x_size 0 --y_size 0`, current code defaults to `1x1` execution.

## Test Catalog
| Test ID | Name | Purpose | Dispatches Kernels |
| :-- | :-- | :-- | :-- |
| `0` | Empty Kernel Launch | Baseline dispatch overhead | Yes |
| `1` | ComputeMM | Full matmul path with optional DRAM and TTNN data transforms | Yes |
| `2` | SubDeviceMM | Split grid into core groups and dispatch independent partitions | Yes |
| `3` | HostPipelineComputeMM | Host-only two-tensor transform + DRAM write/read loop | No |
| `4` | HostPipelineEmpty | Host-only single-tensor baseline pipeline | No |

## CLI Reference
### Core selection and runtime
| Option | Default | Notes |
| :-- | :-- | :-- |
| `--test <0..5>` | `5` | `5` is invalid sentinel; use `0..4` for real tests. |
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

## Architecture and Grid Notes
- Always keep grid sizes within the device logical compute grid.
- Sub-device mode (`--test 2`) requires `y_size >= core_groups`.
- Harvested devices can expose fewer usable rows/cols than max silicon layout.

## Troubleshooting
### Invalid test id
Use `--test` in the `0..4` range.

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
