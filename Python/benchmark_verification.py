from dataclasses import dataclass

import torch


@dataclass
class VerificationResult:
    pcc: float
    relative_error_mean: float
    relative_error_max: float
    relative_error_l2: float
    mae: float
    rmse: float
    max_abs_error: float


def _flatten_pair(golden: torch.Tensor, measured: torch.Tensor):
    if golden.shape != measured.shape:
        raise ValueError(f"Shape mismatch: golden={tuple(golden.shape)} measured={tuple(measured.shape)}")

    golden_f = golden.detach().to(torch.float64).flatten()
    measured_f = measured.detach().to(torch.float64).flatten()
    return golden_f, measured_f


def _compute_pcc(golden_f: torch.Tensor, measured_f: torch.Tensor, eps: float) -> float:
    golden_centered = golden_f - torch.mean(golden_f)
    measured_centered = measured_f - torch.mean(measured_f)

    golden_norm = torch.linalg.vector_norm(golden_centered, ord=2)
    measured_norm = torch.linalg.vector_norm(measured_centered, ord=2)

    if golden_norm <= eps and measured_norm <= eps:
        max_abs_diff = torch.max(torch.abs(golden_f - measured_f)).item()
        return 1.0 if max_abs_diff <= eps else 0.0

    if golden_norm <= eps or measured_norm <= eps:
        return 0.0

    numerator = torch.dot(golden_centered, measured_centered)
    denominator = golden_norm * measured_norm
    return (numerator / denominator).item()


def compute_metrics(golden: torch.Tensor, measured: torch.Tensor, eps: float = 1e-8) -> VerificationResult:
    golden_f, measured_f = _flatten_pair(golden, measured)

    if golden_f.numel() == 0:
        return VerificationResult(
            pcc=1.0,
            relative_error_mean=0.0,
            relative_error_max=0.0,
            relative_error_l2=0.0,
            mae=0.0,
            rmse=0.0,
            max_abs_error=0.0,
        )

    abs_error = torch.abs(measured_f - golden_f)
    max_abs_error = torch.max(abs_error).item()
    mae = torch.mean(abs_error).item()
    rmse = torch.sqrt(torch.mean((measured_f - golden_f) ** 2)).item()

    relative_error = torch.abs(measured_f - golden_f) / (torch.abs(golden_f) + eps)
    rel_mean = torch.mean(relative_error).item()
    rel_max = torch.max(relative_error).item()
    rel_l2 = (
        torch.linalg.vector_norm(measured_f - golden_f, ord=2)
        / (torch.linalg.vector_norm(golden_f, ord=2) + eps)
    ).item()

    pcc = _compute_pcc(golden_f, measured_f, eps)

    return VerificationResult(
        pcc=pcc,
        relative_error_mean=rel_mean,
        relative_error_max=rel_max,
        relative_error_l2=rel_l2,
        mae=mae,
        rmse=rmse,
        max_abs_error=max_abs_error,
    )


def print_visual_comparison(golden: torch.Tensor, measured: torch.Tensor, count: int = 12):
    golden_f, measured_f = _flatten_pair(golden, measured)
    count = min(count, golden_f.numel())

    print(f"[Verify] Visual check ({count} flattened values)")
    for index in range(count):
        g = golden_f[index].item()
        m = measured_f[index].item()
        diff = m - g
        print(f"  idx={index:02d}  golden={g:+.6f}  tt={m:+.6f}  diff={diff:+.6f}")


def verify_and_log(
    path_name: str,
    shape_label: str,
    golden: torch.Tensor,
    measured: torch.Tensor,
    visual_count: int = 12,
):
    metrics = compute_metrics(golden, measured)
    print(
        f"[Verify][{path_name}][{shape_label}] "
        f"PCC={metrics.pcc:.8f} "
        f"REL_MEAN={metrics.relative_error_mean:.6e} "
        f"REL_MAX={metrics.relative_error_max:.6e} "
        f"REL_L2={metrics.relative_error_l2:.6e} "
        f"MAE={metrics.mae:.6e} "
        f"RMSE={metrics.rmse:.6e} "
        f"MAX_ABS={metrics.max_abs_error:.6e}"
    )
    print_visual_comparison(golden, measured, count=visual_count)
    return metrics
