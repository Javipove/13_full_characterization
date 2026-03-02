import argparse
import time
from dataclasses import dataclass

import torch
import ttnn

from memory_budget import detect_dram_capacity_bytes, estimate_and_account_path_usage, log_memory_report
from benchmark_verification import verify_and_log


@dataclass
class BenchmarkResult:
    path_name: str
    shape: tuple[int, int]
    avg_total_ms: float
    avg_prepare_ms: float
    avg_matmul_ms: float
    avg_post_ms: float


def synchronize(device):
    ttnn.synchronize_device(device)


def to_device_legacy_compatible(host_tensor, device, memory_config):
    try:
        return ttnn.to_device(host_tensor, device=device, memory_config=memory_config)
    except TypeError:
        return ttnn.to_device(host_tensor, device=device)


def from_torch_legacy_compatible(host_tensor, dtype, layout, device, memory_config):
    try:
        return ttnn.from_torch(host_tensor, dtype=dtype, layout=layout, device=device, memory_config=memory_config)
    except TypeError:
        tensor = ttnn.from_torch(host_tensor, dtype=dtype, layout=layout)
        return to_device_legacy_compatible(tensor, device, memory_config)


def host_tiling_upload_legacy_compatible(host_tensor, dtype, device, memory_config):
    tiled_on_host = ttnn.from_torch(host_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    return to_device_legacy_compatible(tiled_on_host, device, memory_config)


def to_layout_legacy_compatible(device_tensor, layout):
    if hasattr(ttnn, "to_layout"):
        return ttnn.to_layout(device_tensor, layout)
    raise RuntimeError("This TTNN build does not expose to_layout; cannot run device tiling path.")


def resolve_dtype(dtype_name: str):
    if dtype_name == "bfloat16":
        return ttnn.bfloat16
    if dtype_name == "bfloat8_b":
        return ttnn.bfloat8_b
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def make_host_inputs(m: int, k: int, n: int):
    a = torch.randn((1, 1, m, k), dtype=torch.bfloat16)
    b = torch.randn((1, 1, k, n), dtype=torch.bfloat16)
    return a, b


def run_host_tiling_path(
    device,
    m: int,
    k: int,
    n: int,
    iterations: int,
    memory_config,
    enable_verify: bool,
    visual_count: int,
    dtype,
):
    totals, prepares, matmuls, posts = [], [], [], []

    for iter_idx in range(iterations):
        a_host, b_host = make_host_inputs(m, k, n)

        t0 = time.perf_counter()
        p0 = time.perf_counter()
        a_dev = host_tiling_upload_legacy_compatible(a_host, dtype, device, memory_config)
        b_dev = host_tiling_upload_legacy_compatible(b_host, dtype, device, memory_config)
        synchronize(device)
        p1 = time.perf_counter()

        m0 = time.perf_counter()
        out_dev = ttnn.matmul(a_dev, b_dev, dtype=dtype)
        synchronize(device)
        m1 = time.perf_counter()

        o0 = time.perf_counter()
        measured = ttnn.to_torch(out_dev)
        synchronize(device)
        o1 = time.perf_counter()
        t1 = time.perf_counter()

        if enable_verify and iter_idx == 0:
            golden = torch.matmul(a_host.to(torch.float32), b_host.to(torch.float32)).to(torch.bfloat16)
            verify_and_log(
                path_name="host_tiling_host_untiling",
                shape_label=f"{m}x{k}x{n}",
                golden=golden,
                measured=measured,
                visual_count=visual_count,
            )

        totals.append((t1 - t0) * 1e3)
        prepares.append((p1 - p0) * 1e3)
        matmuls.append((m1 - m0) * 1e3)
        posts.append((o1 - o0) * 1e3)

    return BenchmarkResult(
        path_name="host_tiling_host_untiling",
        shape=(m, n),
        avg_total_ms=sum(totals) / len(totals),
        avg_prepare_ms=sum(prepares) / len(prepares),
        avg_matmul_ms=sum(matmuls) / len(matmuls),
        avg_post_ms=sum(posts) / len(posts),
    )


def run_device_tiling_path(
    device,
    m: int,
    k: int,
    n: int,
    iterations: int,
    memory_config,
    enable_verify: bool,
    visual_count: int,
    dtype,
):
    row_major = ttnn.ROW_MAJOR_LAYOUT
    tile = ttnn.TILE_LAYOUT

    totals, prepares, matmuls, posts = [], [], [], []

    for iter_idx in range(iterations):
        a_host, b_host = make_host_inputs(m, k, n)

        t0 = time.perf_counter()
        p0 = time.perf_counter()
        a_rm = from_torch_legacy_compatible(a_host, dtype, row_major, device, memory_config)
        b_rm = from_torch_legacy_compatible(b_host, dtype, row_major, device, memory_config)
        a_dev = to_layout_legacy_compatible(a_rm, tile)
        b_dev = to_layout_legacy_compatible(b_rm, tile)
        synchronize(device)
        p1 = time.perf_counter()

        m0 = time.perf_counter()
        out_dev = ttnn.matmul(a_dev, b_dev, dtype=dtype)
        synchronize(device)
        m1 = time.perf_counter()

        o0 = time.perf_counter()
        out_rm = to_layout_legacy_compatible(out_dev, row_major)
        measured = ttnn.to_torch(out_rm)
        synchronize(device)
        o1 = time.perf_counter()
        t1 = time.perf_counter()

        if enable_verify and iter_idx == 0:
            golden = torch.matmul(a_host.to(torch.float32), b_host.to(torch.float32)).to(torch.bfloat16)
            verify_and_log(
                path_name="device_tiling_device_untiling",
                shape_label=f"{m}x{k}x{n}",
                golden=golden,
                measured=measured,
                visual_count=visual_count,
            )

        totals.append((t1 - t0) * 1e3)
        prepares.append((p1 - p0) * 1e3)
        matmuls.append((m1 - m0) * 1e3)
        posts.append((o1 - o0) * 1e3)

    return BenchmarkResult(
        path_name="device_tiling_device_untiling",
        shape=(m, n),
        avg_total_ms=sum(totals) / len(totals),
        avg_prepare_ms=sum(prepares) / len(prepares),
        avg_matmul_ms=sum(matmuls) / len(matmuls),
        avg_post_ms=sum(posts) / len(posts),
    )


def print_result(result: BenchmarkResult):
    m, n = result.shape
    print(f"[{result.path_name}] shape={m}x{n}")
    print(f"  avg_total_ms   : {result.avg_total_ms:.3f}")
    print(f"  avg_prepare_ms : {result.avg_prepare_ms:.3f}")
    print(f"  avg_matmul_ms  : {result.avg_matmul_ms:.3f}")
    print(f"  avg_post_ms    : {result.avg_post_ms:.3f}")


def parse_args():
    parser = argparse.ArgumentParser(description="TTNN matmul benchmark (legacy-compatible): host vs device tiling paths")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=["128x128x128", "4096x4096x4096"],
        help="Each shape as MxKxN (must be multiples of 32 for TILE_LAYOUT).",
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--memory-config", choices=["dram", "l1"], default="dram")
    parser.add_argument("--tiling-path", choices=["host", "accelerator", "both"], default="both")
    parser.add_argument("--dtype", choices=["bfloat16", "bfloat8_b"], default="bfloat16")
    parser.add_argument("--verify", action="store_true", help="Enable correctness checks (PCC/relative error + visual sample)")
    parser.add_argument("--visual-count", type=int, default=12)
    parser.add_argument("--dram-capacity-gb", type=float, default=None, help="Optional override; default is dynamic device query")
    parser.add_argument("--dram-fallback-gb", type=float, default=12.0, help="Fallback if dynamic query is unavailable")
    parser.add_argument("--usable-threshold-ratio", type=float, default=0.90, help="Used for informational usage percentages only")
    parser.add_argument("--runtime-overhead-ratio", type=float, default=0.15, help="Safety factor applied to estimated memory")
    return parser.parse_args()


def parse_shape(spec: str):
    parts = spec.lower().split("x")
    if len(parts) != 3:
        raise ValueError(f"Invalid shape '{spec}', expected MxKxN")
    m, k, n = map(int, parts)
    for dim in (m, k, n):
        if dim % 32 != 0:
            raise ValueError(f"Dimension {dim} is not multiple of 32. Use TILE_LAYOUT-compatible dimensions.")
    return m, k, n


def main():
    args = parse_args()
    memory_config = ttnn.DRAM_MEMORY_CONFIG if args.memory_config == "dram" else ttnn.L1_MEMORY_CONFIG
    dtype = resolve_dtype(args.dtype)

    print("[Init] Opening device")
    device = ttnn.open_device(device_id=args.device_id)
    capacity_bytes, capacity_source = detect_dram_capacity_bytes(
        device,
        override_gb=args.dram_capacity_gb,
        fallback_gb=args.dram_fallback_gb,
    )
    print(f"[Init] DRAM capacity source: {capacity_source}, total={capacity_bytes} bytes")

    try:
        for shape in args.shapes:
            m, k, n = parse_shape(shape)
            if args.memory_config == "l1" and max(m, k, n) >= 2048:
                print("[Warn] Large shapes in L1 may cause OOM. Recommended: --memory-config dram")

            if args.dtype == "bfloat8_b" and args.tiling_path in ("accelerator", "both"):
                print("[Warn] bfloat8_b is TILE-layout oriented and not compatible with ROW_MAJOR staging path. Skipping accelerator path.")

            print(f"\n[Run] Shape MxKxN = {m}x{k}x{n}, iterations={args.iterations}, dtype={args.dtype}")
            host_path = None
            dev_path = None

            if args.tiling_path in ("host", "both"):
                host_memory_report = estimate_and_account_path_usage(
                    path_name="host_tiling_host_untiling",
                    m=m,
                    k=k,
                    n=n,
                    dtype_name=args.dtype,
                    capacity_bytes=capacity_bytes,
                    usable_threshold_ratio=args.usable_threshold_ratio,
                    runtime_overhead_ratio=args.runtime_overhead_ratio,
                )
                host_memory_report.capacity_source = capacity_source
                log_memory_report(host_memory_report, f"{m}x{k}x{n}")
                host_path = run_host_tiling_path(
                    device,
                    m,
                    k,
                    n,
                    args.iterations,
                    memory_config,
                    args.verify,
                    args.visual_count,
                    dtype,
                )
                print_result(host_path)

            if args.tiling_path in ("accelerator", "both") and args.dtype != "bfloat8_b":
                device_memory_report = estimate_and_account_path_usage(
                    path_name="device_tiling_device_untiling",
                    m=m,
                    k=k,
                    n=n,
                    dtype_name=args.dtype,
                    capacity_bytes=capacity_bytes,
                    usable_threshold_ratio=args.usable_threshold_ratio,
                    runtime_overhead_ratio=args.runtime_overhead_ratio,
                )
                device_memory_report.capacity_source = capacity_source
                log_memory_report(device_memory_report, f"{m}x{k}x{n}")
                dev_path = run_device_tiling_path(
                    device,
                    m,
                    k,
                    n,
                    args.iterations,
                    memory_config,
                    args.verify,
                    args.visual_count,
                    dtype,
                )
                print_result(dev_path)

            if host_path is not None and dev_path is not None:
                delta = dev_path.avg_total_ms - host_path.avg_total_ms
                print(f"  delta_total_ms(device-host): {delta:.3f}")
    finally:
        print("\n[Shutdown] Closing device")
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
