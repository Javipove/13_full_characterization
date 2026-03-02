from dataclasses import dataclass


BF16_BYTES = 2
TILE_HW = 32
TILE_ELEMS = TILE_HW * TILE_HW
BFP8_BLOCK_VALUES = 16
BFP8_BLOCK_BYTES = 17

# Official tile-byte rules from tt-metal DataFormat conventions:
# - bfloat16 tile = 32x32 * 2 bytes = 2048 B
# - bfloat8_b tile = 1088 B (1024 mantissa bytes + 64 exponent bytes)
TILE_SIZE_BYTES_BY_DTYPE = {
    "bfloat16": TILE_ELEMS * BF16_BYTES,
    "bfloat8_b": 1088,
}

ROW_MAJOR_BYTES_BY_DTYPE = {
    "bfloat16": 2,
}


@dataclass
class MemoryBudgetReport:
    path_name: str
    estimated_total_bytes: int
    accounted_usage_bytes: int
    logical_tensor_total_bytes: int
    tile_padded_tensor_total_bytes: int
    input_tensors_bytes: int
    intermediate_tensors_bytes: int
    final_result_tensors_bytes: int
    tiling_untiling_occupancy_bytes: int
    matmul_occupancy_bytes: int
    peak_pipeline_occupancy_bytes: int
    padding_overhead_bytes: int
    capacity_bytes: int
    capacity_source: str
    usable_threshold_ratio: float

    @property
    def estimated_pct_capacity(self) -> float:
        return 100.0 * self.estimated_total_bytes / self.capacity_bytes

    @property
    def accounted_pct_capacity(self) -> float:
        return 100.0 * self.accounted_usage_bytes / self.capacity_bytes

    @property
    def estimated_pct_usable(self) -> float:
        return 100.0 * self.estimated_total_bytes / (self.capacity_bytes * self.usable_threshold_ratio)

    @property
    def accounted_pct_usable(self) -> float:
        return 100.0 * self.accounted_usage_bytes / (self.capacity_bytes * self.usable_threshold_ratio)


def tensor_bytes(shape: tuple[int, ...], element_size_bytes: int = BF16_BYTES) -> int:
    count = 1
    for dim in shape:
        count *= dim
    return count * element_size_bytes


def round_up(value: int, multiple: int) -> int:
    if multiple <= 0:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def tile_padded_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    if len(shape) < 2:
        return shape

    padded = list(shape)
    padded[-2] = round_up(padded[-2], TILE_HW)
    padded[-1] = round_up(padded[-1], TILE_HW)
    return tuple(padded)


def tensor_bytes_tile_layout(shape: tuple[int, ...], element_size_bytes: int = BF16_BYTES) -> int:
    return tensor_bytes(tile_padded_shape(shape), element_size_bytes)


def tensor_bytes_tile_layout_by_dtype(shape: tuple[int, ...], dtype_name: str) -> int:
    padded = tile_padded_shape(shape)
    if len(padded) < 2:
        raise ValueError("Tile-layout sizing requires rank >= 2")

    tile_bytes = TILE_SIZE_BYTES_BY_DTYPE.get(dtype_name)
    if tile_bytes is None:
        raise ValueError(f"Unsupported dtype for tile sizing: {dtype_name}")

    total_tiles = 1
    for dim in padded[:-2]:
        total_tiles *= dim
    total_tiles *= padded[-2] // TILE_HW
    total_tiles *= padded[-1] // TILE_HW
    return total_tiles * tile_bytes


def tensor_bytes_row_major_by_dtype(shape: tuple[int, ...], dtype_name: str) -> int:
    if dtype_name == "bfloat8_b":
        raise ValueError("bfloat8_b row-major staging is not supported in TTNN memory model")

    elem_bytes = ROW_MAJOR_BYTES_BY_DTYPE.get(dtype_name)
    if elem_bytes is None:
        raise ValueError(f"Unsupported dtype for row-major sizing: {dtype_name}")
    return tensor_bytes(shape, elem_bytes)


def tensor_bytes_logical_by_dtype(shape: tuple[int, ...], dtype_name: str) -> int:
    # Logical storage model for academic reporting:
    # - bfloat16: dense 2-byte per element payload.
    # - bfloat8_b: block-floating payload where 16 mantissas share one exponent,
    #   so memory is modeled as ceil(num_values/16) * 17 bytes.
    total_values = 1
    for dim in shape:
        total_values *= dim

    if dtype_name == "bfloat16":
        return total_values * BF16_BYTES
    if dtype_name == "bfloat8_b":
        blocks = (total_values + BFP8_BLOCK_VALUES - 1) // BFP8_BLOCK_VALUES
        return blocks * BFP8_BLOCK_BYTES
    raise ValueError(f"Unsupported dtype for logical sizing: {dtype_name}")


def _base_matmul_bytes(m: int, k: int, n: int, dtype_name: str) -> tuple[int, int, int]:
    a_bytes = tensor_bytes_logical_by_dtype((1, 1, m, k), dtype_name)
    b_bytes = tensor_bytes_logical_by_dtype((1, 1, k, n), dtype_name)
    c_bytes = tensor_bytes_logical_by_dtype((1, 1, m, n), dtype_name)
    return a_bytes, b_bytes, c_bytes


def _base_matmul_row_major_bytes(m: int, k: int, n: int, dtype_name: str) -> tuple[int, int, int]:
    a_bytes = tensor_bytes_row_major_by_dtype((1, 1, m, k), dtype_name)
    b_bytes = tensor_bytes_row_major_by_dtype((1, 1, k, n), dtype_name)
    c_bytes = tensor_bytes_row_major_by_dtype((1, 1, m, n), dtype_name)
    return a_bytes, b_bytes, c_bytes


def _base_matmul_tile_bytes(m: int, k: int, n: int, dtype_name: str) -> tuple[int, int, int]:
    a_tile_bytes = tensor_bytes_tile_layout_by_dtype((1, 1, m, k), dtype_name)
    b_tile_bytes = tensor_bytes_tile_layout_by_dtype((1, 1, k, n), dtype_name)
    c_tile_bytes = tensor_bytes_tile_layout_by_dtype((1, 1, m, n), dtype_name)
    return a_tile_bytes, b_tile_bytes, c_tile_bytes


def _to_int_or_none(value):
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _call_noarg_if_exists(obj, name: str):
    attr = getattr(obj, name, None)
    if attr is None:
        return None
    try:
        return attr() if callable(attr) else attr
    except Exception:
        return None


def detect_dram_capacity_bytes(device, override_gb: float | None = None, fallback_gb: float = 12.0) -> tuple[int, str]:
    if override_gb is not None:
        return int(override_gb * (1024**3)), "override"

    size_per_channel = _to_int_or_none(_call_noarg_if_exists(device, "dram_size_per_channel"))
    num_channels = _to_int_or_none(_call_noarg_if_exists(device, "num_dram_channels"))

    if size_per_channel is not None and num_channels is not None and size_per_channel > 0 and num_channels > 0:
        return size_per_channel * num_channels, "device.dram_size_per_channel*num_dram_channels"

    dram_grid = _call_noarg_if_exists(device, "dram_grid_size")
    grid_x = _to_int_or_none(getattr(dram_grid, "x", None))
    grid_y = _to_int_or_none(getattr(dram_grid, "y", None))

    if size_per_channel is not None and size_per_channel > 0 and grid_x is not None and grid_x > 0:
        banks = grid_x * (grid_y if grid_y is not None and grid_y > 0 else 1)
        return size_per_channel * banks, "device.dram_size_per_channel*dram_grid_size"

    allocator = _call_noarg_if_exists(device, "allocator")
    if allocator is not None:
        bank_size = _to_int_or_none(_call_noarg_if_exists(allocator, "get_bank_size"))
        num_banks = _to_int_or_none(_call_noarg_if_exists(allocator, "get_num_banks"))
        if bank_size is not None and num_banks is not None and bank_size > 0 and num_banks > 0:
            return bank_size * num_banks, "device.allocator"

    return int(fallback_gb * (1024**3)), "fallback"


def estimate_and_account_path_usage(
    path_name: str,
    m: int,
    k: int,
    n: int,
    dtype_name: str,
    capacity_bytes: int,
    usable_threshold_ratio: float,
    runtime_overhead_ratio: float = 0.15,
) -> MemoryBudgetReport:
    a_bytes, b_bytes, c_bytes = _base_matmul_bytes(m, k, n, dtype_name)
    a_tile_bytes, b_tile_bytes, c_tile_bytes = _base_matmul_tile_bytes(m, k, n, dtype_name)

    # Logical tensor occupancy is the mathematically expected payload size without TT tile padding.
    # This is useful for understanding model-level memory requirements independent of hardware layout.
    logical_tensor_total = a_bytes + b_bytes + c_bytes

    # Tile-padded tensor occupancy applies TTNN TILE_LAYOUT constraints (32x32 padded last two dims).
    # This is the closest static approximation to real tensor storage when tensors are in TILE_LAYOUT.
    tile_padded_tensor_total = a_tile_bytes + b_tile_bytes + c_tile_bytes

    if path_name == "host_tiling_host_untiling":
        # Host tiling path assumptions:
        # - Inputs are tiled on host and uploaded, so DRAM stores tile-shaped A and B.
        # - Matmul output is tile-shaped C on device before host readback.
        # - No explicit additional device-side conversion tensors are created in this path.
        input_tensors = a_tile_bytes + b_tile_bytes
        intermediate_tensors = 0
        final_result_tensors = c_tile_bytes
        accounted = input_tensors + intermediate_tensors + final_result_tensors
        baseline = logical_tensor_total

        # Host-tiling path split:
        # - No accelerator-side to_layout/untilize pipeline is modeled.
        # - Matmul occupancy uses the resident tiled tensors only.
        tiling_untiling_occupancy = 0
        matmul_occupancy = a_tile_bytes + b_tile_bytes + c_tile_bytes
        peak_pipeline_occupancy = accounted
    elif path_name == "device_tiling_device_untiling":
        if dtype_name == "bfloat8_b":
            raise ValueError("device_tiling_device_untiling assumes row-major staging and is not valid for bfloat8_b")

        a_row_bytes, b_row_bytes, c_row_bytes = _base_matmul_row_major_bytes(m, k, n, dtype_name)

        # Accelerator tiling path assumptions:
        # - Row-major A/B uploads are present in DRAM.
        # - Device-side to_layout creates tile-layout A/B tensors (counted as intermediates).
        # - Matmul produces tile-layout C, and untiling creates row-major C for host transfer.
        # - We explicitly separate final results from intermediates for user-facing occupancy analysis.
        input_tensors = a_row_bytes + b_row_bytes
        intermediate_tensors = a_tile_bytes + b_tile_bytes
        final_result_tensors = c_tile_bytes + c_row_bytes
        accounted = input_tensors + intermediate_tensors + final_result_tensors
        baseline = (a_row_bytes + b_row_bytes) + (a_row_bytes + b_row_bytes) + (c_row_bytes + c_row_bytes)

        # Accelerator conversion occupancy (tiling/untiling) and matmul occupancy are reported separately.
        # These are two different "views" of memory pressure and are intentionally non-additive:
        # - tiling_untiling_occupancy: buffers associated with row-major <-> tile conversion pipeline.
        # - matmul_occupancy: core matmul resident tensors in tile layout (A_tile, B_tile, C_tile).
        tiling_untiling_occupancy = input_tensors + intermediate_tensors + final_result_tensors
        matmul_occupancy = a_tile_bytes + b_tile_bytes + c_tile_bytes
        # Peak pipeline occupancy (conservative) models full accelerator path residency where
        # conversion and compute-related buffers can coexist before host readback is consumed.
        peak_pipeline_occupancy = accounted
    else:
        raise ValueError(f"Unsupported path_name: {path_name}")

    # Padding overhead isolates additional bytes caused only by tile alignment requirements.
    # Baseline is path-specific and keeps the same number of logical buffers but without tile padding.
    padding_overhead = max(0, accounted - baseline)

    # Estimated total adds a runtime overhead factor to account for allocator/runtime transients.
    # This mirrors the practical recommendation from TT docs: tensor-bytes alone underestimates real peak usage.
    estimated = int(accounted * (1.0 + runtime_overhead_ratio))

    return MemoryBudgetReport(
        path_name=path_name,
        estimated_total_bytes=estimated,
        accounted_usage_bytes=accounted,
        logical_tensor_total_bytes=logical_tensor_total,
        tile_padded_tensor_total_bytes=tile_padded_tensor_total,
        input_tensors_bytes=input_tensors,
        intermediate_tensors_bytes=intermediate_tensors,
        final_result_tensors_bytes=final_result_tensors,
        tiling_untiling_occupancy_bytes=tiling_untiling_occupancy,
        matmul_occupancy_bytes=matmul_occupancy,
        peak_pipeline_occupancy_bytes=peak_pipeline_occupancy,
        padding_overhead_bytes=padding_overhead,
        capacity_bytes=capacity_bytes,
        capacity_source="runtime",
        usable_threshold_ratio=usable_threshold_ratio,
    )


def bytes_to_human(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    unit_index = 0
    while value >= 1024.0 and unit_index < len(units) - 1:
        value /= 1024.0
        unit_index += 1
    return f"{value:.2f} {units[unit_index]}"


def log_memory_report(report: MemoryBudgetReport, shape_label: str):
    print(
        f"[Memory][{report.path_name}][{shape_label}] "
        f"CAP_SOURCE={report.capacity_source} "
        f"LOGICAL_TENSORS={bytes_to_human(report.logical_tensor_total_bytes)} "
        f"TILE_PADDED_TENSORS={bytes_to_human(report.tile_padded_tensor_total_bytes)} "
        f"INPUTS={bytes_to_human(report.input_tensors_bytes)} "
        f"INTERMEDIATES={bytes_to_human(report.intermediate_tensors_bytes)} "
        f"FINAL_RESULTS={bytes_to_human(report.final_result_tensors_bytes)} "
        f"TILING_UNTILING_OCC={bytes_to_human(report.tiling_untiling_occupancy_bytes)} "
        f"MATMUL_OCC={bytes_to_human(report.matmul_occupancy_bytes)} "
        f"PEAK_PIPELINE_OCC={bytes_to_human(report.peak_pipeline_occupancy_bytes)} "
        f"EST_TOTAL={bytes_to_human(report.estimated_total_bytes)} ({report.estimated_pct_capacity:.2f}% cap, {report.estimated_pct_usable:.2f}% usable) "
        f"ACTUAL_ACCOUNTED={bytes_to_human(report.accounted_usage_bytes)} ({report.accounted_pct_capacity:.2f}% cap, {report.accounted_pct_usable:.2f}% usable) "
        f"PADDING_OVERHEAD={bytes_to_human(report.padding_overhead_bytes)}"
    )
