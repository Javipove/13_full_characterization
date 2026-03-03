# Python Benchmarks Guide

<p>
  Compact public guide for the TTNN matmul benchmarking examples in this folder.
</p>

<h2>Files and Purpose</h2>

<ul>
  <li><code>matmul_tiling_benchmark_new_api.py</code>: modern TTNN API benchmark for host/accelerator tiling paths, timing, verification, and DRAM occupancy output.</li>
  <li><code>matmul_tiling_benchmark_legacy_api.py</code>: legacy-compatible benchmark with fallback wrappers for older signatures.</li>
  <li><code>benchmark_verification.py</code>: validation metrics (PCC, relative errors, MAE/RMSE, max-abs) and visual sample comparison.</li>
  <li><code>memory_budget.py</code>: dynamic accelerator DRAM capacity detection and occupancy estimation (logical, tile-padded, conversion, matmul, peak).</li>
</ul>

<h2>Quick Start</h2>

```bash
python matmul_tiling_benchmark_new_api.py --iterations 5
python matmul_tiling_benchmark_legacy_api.py --iterations 5
```

<details>
  <summary><strong>CLI Parameters (both benchmark scripts)</strong></summary>

| Parameter | Type | Default | Purpose |
| --- | ---: | ---: | --- |
| `--iterations` | int | `5` | Iterations per shape/path. |
| `--shapes` | list[str] | `128x128x128 4096x4096x4096` | Shapes in `MxKxN`; each dimension must be multiple of 32. |
| `--device-id` | int | `0` | Accelerator device ID. |
| `--memory-config` | `dram/l1` | `dram` | Tensor memory placement preference. |
| `--tiling-path` | `host/accelerator/both` | `both` | Select conversion path(s) to benchmark. |
| `--dtype` | `bfloat16/bfloat8_b` | `bfloat16` | Matmul dtype in TTNN. |
| `--verify` | flag | off | Enable correctness metrics and sample comparison. |
| `--visual-count` | int | `12` | Number of visual sample values. |
| `--dram-capacity-gb` | float | unset | Optional manual override for accelerator DRAM capacity. |
| `--dram-fallback-gb` | float | `12.0` | Fallback capacity if runtime query is unavailable. |
| `--usable-threshold-ratio` | float | `0.90` | Informational occupancy percentage denominator. |
| `--runtime-overhead-ratio` | float | `0.15` | Conservative factor used in `EST_TOTAL`. |

</details>

<details>
  <summary><strong>Common Commands</strong></summary>

```bash
# New API default run
python matmul_tiling_benchmark_new_api.py --iterations 5 --memory-config dram

# Legacy API default run
python matmul_tiling_benchmark_legacy_api.py --iterations 5 --memory-config dram

# Host-only / accelerator-only path
python matmul_tiling_benchmark_new_api.py --tiling-path host
python matmul_tiling_benchmark_new_api.py --tiling-path accelerator

# Enable verification metrics
python matmul_tiling_benchmark_new_api.py --verify --visual-count 12

# Custom shapes
python matmul_tiling_benchmark_new_api.py --iterations 3 --shapes 256x256x256 2048x2048x2048

# bfloat8_b (host path)
python matmul_tiling_benchmark_new_api.py --dtype bfloat8_b --tiling-path host
```

</details>

<details>
  <summary><strong>Reported Outputs</strong></summary>

<ul>
  <li>Timing: <code>avg_prepare_ms</code>, <code>avg_matmul_ms</code>, <code>avg_post_ms</code>, <code>avg_total_ms</code></li>
  <li>Verification (<code>--verify</code>): PCC, relative errors, MAE, RMSE, max abs, plus visual sample values</li>
  <li>DRAM occupancy (informational, non-blocking): logical size, tile-padded size, split occupancy (<code>INPUTS</code>, <code>INTERMEDIATES</code>, <code>FINAL_RESULTS</code>), conversion occupancy, matmul occupancy, peak pipeline occupancy, estimated total and percentages</li>
</ul>

</details>

<h2>Old vs New API (Non-Compatibility Rationale)</h2>

<ul>
  <li><strong>New API script</strong> (<code>matmul_tiling_benchmark_new_api.py</code>): uses current TTNN signatures directly.</li>
  <li><strong>Legacy API script</strong> (<code>matmul_tiling_benchmark_legacy_api.py</code>): includes compatibility wrappers for older signatures.</li>
</ul>

<p><strong>Main reasons compatibility diverges:</strong></p>
<ul>
  <li>Signature drift across versions (<code>from_torch</code>, <code>to_device</code>, optional arguments).</li>
  <li>Layout conversion support differences (<code>to_layout</code> availability/behavior).</li>
  <li><code>bfloat8_b</code> is tile-oriented; row-major accelerator staging is intentionally excluded for correctness.</li>
  <li>Runtime/allocator internals evolve across releases, changing buffer-lifetime assumptions and memory accounting behavior.</li>
</ul>

<p><strong>Practical guidance:</strong> use the new script on modern stacks, and the legacy script only for older TT-Metal/TTNN environments.</p>
