# Device-Side Tiling, BFP8 Packing, and Untiling in ttnn/tt-metal (Wormhole)

## Architecture Overview

The Wormhole device (and Blackhole) uses a **three-engine pipeline** for every compute kernel: **Unpacker → SFPU Math → Packer**. Crucially, for all BFP8 ↔ float conversions, the unpacker/packer hardware handles the format conversion natively — **no SFPU kernel is needed**.

```mermaid
graph TD
    "Host (CPU)" --> "Row-Major BF16 tensor in DRAM"
    "Row-Major BF16 tensor in DRAM" --> "ttnn.tilize / ttnn.tilize_with_val_padding"
    "ttnn.tilize / ttnn.tilize_with_val_padding" --> "TILE_LAYOUT tensor (BF16 or BFP8_B)"
    "TILE_LAYOUT tensor (BF16 or BFP8_B)" --> "ttnn.matmul"
    "ttnn.matmul" --> "TILE_LAYOUT BF16 output"
    "TILE_LAYOUT BF16 output" --> "ttnn.untilize / ttnn.untilize_with_unpadding"
    "ttnn.untilize / ttnn.untilize_with_unpadding" --> "Row-Major output"
    "TILE_LAYOUT tensor (BF16 or BFP8_B)" --> "ttnn.typecast"
    "ttnn.typecast" --> "TILE_LAYOUT BFP8_B"
```

---

## 1. On-Device Tiling (ROW_MAJOR → TILE)

### High-Level API: `ttnn.tilize`

The top-level registered operation is declared in `ttnn/cpp/ttnn/operations/data_movement/tilize/tilize.hpp`: [1](#4-0) 

Its Python binding exposes: `ttnn.tilize(input_tensor, *, memory_config=None, dtype=None, use_multicore=True, use_low_perf=False, sub_core_grids=None)`: [2](#4-1) 

The `dtype` argument lets you **simultaneously tilize AND convert to BFP8 in a single on-device pass**.

### Tilize With Padding: `ttnn.tilize_with_val_padding` and `ttnn.tilize_with_zero_padding`

When input dimensions are not already multiples of 32, `ttnn.tilize_with_val_padding` is used: [3](#4-2) 

### High-Level Dispatcher: `ttnn.to_layout`

The recommended unified API is `ttnn.to_layout`, which selects the right variant automatically. When converting a device tensor to `TILE_LAYOUT`:
- If no padding change is needed → calls `ttnn::tilize`
- If `HEIGHT_SHARDED` → calls `ttnn::pad` then `ttnn::tilize`
- Otherwise → calls `ttnn::tilize_with_val_padding` [4](#4-3) 

### Program Factories

The `TilizeDeviceOperation` selects from multiple program factories:
- `TilizeSingleCoreProgramFactory`
- `TilizeMultiCoreInterleavedProgramFactory`
- `TilizeMultiCoreBlockProgramFactory`
- `TilizeMultiCoreShardedProgramFactory`
- `TilizeMultiCoreWidthShardedProgramFactory` [5](#4-4) 

Validation requires input to be `ROW_MAJOR`, on-device, with supported dtypes (BFLOAT16, FLOAT32, UINT32, INT32, UINT16): [6](#4-5) 

### Tilize Compute Kernels

The core compute kernel (`tilize.cpp`) uses the LLK `tilize_init` + `tilize_block` API: [7](#4-6) 

The Wormhole-optimized variant (`tilize_wh.cpp`) uses `tilize_uninit`: [8](#4-7) 

### LLK Compute API (`tilize.h`)

`tilize_init` wires all three engines (UNPACK, MATH, PACK) for the tilize operation: [9](#4-8) 

---

## 2. On-Device BFP8 Packing (FP32/BF16 → BFP8)

### Method A: Combined Tilize + BFP8 Pack via `ttnn.tilize` / `ttnn.to_layout`

Passing `dtype=ttnn.bfloat8_b` to `ttnn.tilize_with_val_padding` performs tilization AND BFP8 packing in a **single device pass**. The device packer hardware handles the BFP8 encoding directly (no SFPU needed): [10](#4-9) 

The program factory sets up input CBs at BF16 format and output CBs at BFP8 format, configuring the packer to convert automatically.

### Method B: `ttnn.typecast` on Already-Tiled Tensor

For a tensor already in `TILE_LAYOUT`, use `ttnn.typecast`: [11](#4-10) 

`typecast_impl` sets `bfp8_pack_precise = true` when output is `BFLOAT8_B`: [12](#4-11) 

### Hardware-Level: Packer Handles BFP8 Natively (No SFPU)

On Wormhole, the LLK typecast header explicitly shows that `Float16_b → Bfp8_b`, `Float32 → Bfp8_b`, and `Bfp8_b → Float32/Float16_b` conversions require **zero SFPU kernel work** — they are handled entirely by the packer/unpacker hardware: [13](#4-12) 

### `bfp8_pack_precise` Flag in Data Format Pipeline

The `get_single_pack_src_format` function controls what the packer reads from DEST when writing BFP8 output. With `bfp8_pack_precise=true` (set automatically by `typecast_impl`) and `fp32_dest_acc_en=true`, the packer reads from `Float32` for maximum precision: [14](#4-13) 

Without `bfp8_pack_precise`, it reads directly from `Bfp8_b` (faster, lower precision): [15](#4-14) 

---

## 3. On-Device Untiling / Unpacking (TILE → ROW_MAJOR)

### `ttnn.untilize`

Declared and registered as: [16](#4-15) 

Python binding: `ttnn.untilize(input_tensor, *, memory_config=None, use_multicore=True, use_pack_untilize=True, sub_core_grids=None)`: [17](#4-16) 

**Critical**: When input dtype is `BFLOAT8_B`, `untilize` **automatically** converts output to `BFLOAT16` — the hardware unpacker performs the BFP8→BF16 expansion with no SFPU: [18](#4-17) 

### `ttnn.untilize_with_unpadding`

Used by `ttnn.to_layout` when converting from `TILE_LAYOUT` back to `ROW_MAJOR` with padding removal: [19](#4-18) 

### `use_pack_untilize` Mode

When `use_pack_untilize=True` (default), the faster `pack_untilize` kernel path is used. This kernel (`pack_untilize.cpp`) uses `pack_untilize_init` + `pack_untilize_block` for higher throughput: [20](#4-19) 

The Wormhole-optimized variant (`pack_untilize_wh.cpp`) uses the same pattern: [21](#4-20) 

The standard untilize kernel (when pack_untilize disabled) uses `untilize_init` + `untilize_block`: [22](#4-21) 

### Untilize Program Factory Selection

The factory selection logic handles multiple scenarios including sharded, sub-core-grid, block, column-parallel, and ND-shard modes: [23](#4-22) 

---

## 4. Complete Operation Table

| ttnn Operation | Device Kernel Used | BFP8 Support | Notes |
|---|---|---|---|
| `ttnn.tilize` | `tilize.cpp` / `tilize_wh.cpp` | Via `dtype=` arg | ROW_MAJOR→TILE on-device |
| `ttnn.tilize_with_val_padding` | `tilize.cpp` | Via `dtype=ttnn.bfloat8_b` | Tilize + pad + optional BFP8 pack |
| `ttnn.tilize_with_zero_padding` | `tilize.cpp` | Via `output_dtype=` | Convenience zero-padding variant |
| `ttnn.untilize` | `untilize.cpp` / `pack_untilize.cpp` / `pack_untilize_wh.cpp` | Auto BFP8→BF16 | TILE→ROW_MAJOR on-device |
| `ttnn.untilize_with_unpadding` | Same as untilize | Auto BFP8→BF16 | Removes pad on untilize |
| `ttnn.to_layout` | Dispatches above | Via `dtype=` arg | Unified high-level API |
| `ttnn.typecast` | Packer/unpacker hardware + SFPU for non-float | Full BFP8 support | Converts dtype of TILE tensor |

---

## 5. Host-Side vs. Device-Side Pipeline Comparison

### Current Host-Side Approach

The host-side `pack_as_bfp8_tiles` (C++) or `_tilize_and_pack_bfp8` (Python/NumPy) computes shared exponents per 16-element block on the CPU, producing `[16 exp words][256 mantissa words]` per tile: [24](#4-23) 

This is used by `tensor_impl.cpp` and `prepare_conv2d_weights.cpp` to pre-pack weights on the host before uploading to device.

The `compute_host_dtype` function in `py_to_tt_tensor.cpp` maps `BFLOAT8_B` target dtype to `FLOAT32` host dtype — i.e., the host converts to float32, **then** the BFP8 packing happens on the host: [25](#4-24) 

**All dtype conversions are currently forced to host** due to known precision and functional bugs (issues #35048, #31406, #23405): [26](#4-25) 

### On-Device Path (When Used)

When a device is provided and dtype still doesn't match after host construction, `py_to_tt_tensor.cpp` performs the conversion on-device: [27](#4-26) 

This calls `set_layout(TILE)` then `ttnn::typecast(output, dst_dtype)`.

### `col_tilize` Mode (Special BFP8 Weight Optimization)

For BFP8 weight matrices, `col_tilize=true` transposes the host buffer so BFP8 shared exponents are grouped **column-wise** (along K dimension) rather than row-wise. This is used for KV-cache-style weight layouts: [28](#4-27) 

---

## 6. Integration into Matmul / LLM Inference Pipeline

### Typical LLM Decode (DRAM-Sharded Weights)

```mermaid
graph TD
    "BF16 Activations (ROW_MAJOR)" --> "ttnn.to_layout(TILE_LAYOUT)"
    "ttnn.to_layout(TILE_LAYOUT)" --> "BF16 Activations (TILE)"
    "BF16 Activations (TILE)" --> "ttnn.matmul (HiFi2 fidelity)"
    "BFP8_B Weights in DRAM (TILE)" --> "ttnn.matmul (HiFi2 fidelity)"
    "ttnn.matmul (HiFi2 fidelity)" --> "BF16 Output (TILE)"
    "BF16 Output (TILE)" --> "ttnn.to_layout(ROW_MAJOR)"
```

- Weights are pre-packed to `BFLOAT8_B + TILE_LAYOUT` on host via `pack_as_bfp8_tiles`
- Activations are tilized on-device via `ttnn.tilize` or `ttnn.to_layout`
- During matmul, the **unpacker hardware** automatically expands BFP8_B tiles from DRAM to BF16 in the DEST register — zero extra kernel overhead
- Output untilized via `ttnn.untilize` (with `use_pack_untilize=True` for ~5× throughput vs. standard untilize path, per measured latencies) [29](#4-28) 

### Fully On-Device Pipeline (Future/Ideal Path)

```mermaid
graph TD
    "BF16 Activations (ROW_MAJOR, DRAM)" --> "ttnn.to_layout(TILE_LAYOUT, dtype=bfloat8_b)"
    "ttnn.to_layout(TILE_LAYOUT, dtype=bfloat8_b)" --> "BFP8_B Activations (TILE)"
    "BFP8_B Activations (TILE)" --> "ttnn.matmul"
    "BFP8_B Weights (TILE)" --> "ttnn.matmul"
    "ttnn.matmul" --> "BF16 Output (TILE)"
    "BF16 Output (TILE)" --> "ttnn.untilize"
    "ttnn.untilize" --> "BF16 Output (ROW_MAJOR)"
```

`ttnn.to_layout(tensor, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)` performs **tilize + BFP8 pack in one pass** via `ttnn::tilize_with_val_padding` with `output_dtype=BFLOAT8_B`. This eliminates the host-round-trip entirely for activation quantization. The packer hardware does the float16_b → BFP8_b conversion natively.

However, this path is currently **disabled by default** due to the `can_exec_ops_on_device` guard returning `false` for `BFLOAT8_B` (issue #35048). The underlying device operations (`ttnn.tilize` with `dtype=bfloat8_b` and `ttnn.typecast`) are fully implemented and working for explicitly invoked paths (e.g., custom model kernels, `test_sharded.py`).

---

## Notes

1. **BFP8 hardware path**: On Wormhole, `Bfp8_b ↔ Float16_b` and `Bfp8_b ↔ Float32` conversions are handled **entirely by the unpacker/packer** — they appear as no-ops at the SFPU level. Only `Bfp8_b ↔ UInt16`, `Bfp8_b ↔ Int32`, etc. require SFPU kernels. [30](#4-29) 

2. **`ttnn.typecast` is registered at top level** (`ttnn::typecast`, not `ttnn::experimental::typecast`) via `register_operation<"ttnn::typecast", ...>()`: [31](#4-30) 

3. **`untilize` auto-converts BFP8**: If input is `BFLOAT8_B`, `ttnn.untilize` always outputs `BFLOAT16` — you cannot stay in BFP8 after untilizing: [32](#4-31) 

4. **Tilize supported input dtypes**: `ttnn.tilize` validates input must be `BFLOAT16`, `FLOAT32`, `UINT32`, `INT32`, or `UINT16`. It does **not** accept `BFLOAT8_B` as input; BFP8 input must be handled via `ttnn.typecast`: [33](#4-32) 

5. **The host packing path** (`pack_as_bfp8_tiles`) will remain the production path for weight pre-loading and any scenario requiring fused buffer layouts (e.g., the `BlitzDecodeWeights` approach for DeepSeek).

### Citations

**File:** ttnn/cpp/ttnn/operations/data_movement/tilize/tilize.hpp (L12-24)
```text
struct ExecuteTilize {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<DataType> output_dtype = std::nullopt,
        bool use_multicore = true,
        bool use_low_perf = false,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
};

}  // namespace operations::data_movement

constexpr auto tilize = ttnn::register_operation<"ttnn::tilize", ttnn::operations::data_movement::ExecuteTilize>();
```

**File:** ttnn/cpp/ttnn/operations/data_movement/tilize/tilize_nanobind.cpp (L17-61)
```cpp
void bind_tilize(nb::module_& mod) {
    const auto* doc = R"doc(
        Changes data layout of input tensor to TILE.

        Input tensor must be on TT accelerator device, in ROW_MAJOR layout, and have BFLOAT16 data type.

        Output tensor will be on TT accelerator device, in TILE layout, and have BFLOAT16 data type.

        Args:
            input_tensor (ttnn.Tensor): the input tensor.

        Keyword Args:
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
            dtype (data type, optional): Data type of the output tensor. Defaults to `None`.
            use_multicore (bool, optional): Whether to use multicore. Defaults to `True`.
            use_low_perf (bool, optional): Use a low performance version that uses less memory. USE ONLY IF ABSOLUTELY NEEDED IN MODELS. Defaults to `False`.
            sub_core_grids (CoreRangeSet, optional): Used to restrict tilize to a set of cores, Defaults to using the entire device

        Returns:
            ttnn.Tensor: the output tensor.
    )doc";

    using OperationType = decltype(ttnn::tilize);
    ttnn::bind_registered_operation(
        mod,
        ttnn::tilize,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const std::optional<MemoryConfig>& memory_config,
               std::optional<DataType> output_dtype,
               bool use_multicore,
               bool use_low_perf,
               const std::optional<CoreRangeSet>& sub_core_grids) {
                return self(input_tensor, memory_config, output_dtype, use_multicore, use_low_perf, sub_core_grids);
            },
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("dtype") = nb::none(),
            nb::arg("use_multicore") = true,
            nb::arg("use_low_perf") = false,
            nb::arg("sub_core_grids") = nb::none()});
}
```

**File:** ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp (L14-50)
```text
struct ExecuteTilizeWithValPadding {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::SmallVector<uint32_t>& output_padded_shape,
        tt::tt_metal::PadValue pad_value,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<DataType> output_dtype = std::nullopt,
        bool use_multicore = true,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Shape& output_padded_shape,
        tt::tt_metal::PadValue pad_value,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<DataType> output_dtype = std::nullopt,
        bool use_multicore = true,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
};

struct ExecuteTilizeWithZeroPadding {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<DataType> output_dtype = std::nullopt,
        bool use_multicore = true,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
};

}  // namespace operations::data_movement

constexpr auto tilize_with_val_padding = ttnn::
    register_operation<"ttnn::tilize_with_val_padding", ttnn::operations::data_movement::ExecuteTilizeWithValPadding>();

constexpr auto tilize_with_zero_padding = ttnn::register_operation<
    "ttnn::tilize_with_zero_padding",
    ttnn::operations::data_movement::ExecuteTilizeWithZeroPadding>();
```

**File:** ttnn/cpp/ttnn/operations/core/to_layout/to_layout_op.cpp (L92-199)
```cpp
    if (tt::tt_metal::is_device_tensor(tensor_arg)) {
        bool use_multicore_untilize = true;
        bool use_multicore_tilize = true;

        if (not requires_padding_change(tensor, layout)) {
            if (layout == ttnn::ROW_MAJOR_LAYOUT) {
                TT_ASSERT(not dtype.has_value(), "dtype cannot be specified when converting to ROW_MAJOR_LAYOUT!");
                return ttnn::untilize(
                    tensor, output_memory_config, use_multicore_untilize, true /*use_pack_untilize*/, sub_core_grids);
            }
            if (layout == ttnn::TILE_LAYOUT) {
                if (tensor.is_sharded()) {
                    const auto tensor_tile = tensor.tensor_spec().tile();
                    uint32_t tile_height = tensor_tile.get_height();
                    uint32_t tile_width = tensor_tile.get_width();
                    const auto shard_shape = get_memory_config(tensor).value().shard_spec().value().shape;
                    if (shard_shape[0] % tile_height != 0 or shard_shape[1] % tile_width != 0) {
                        TT_THROW(
                            "ttnn::to_layout: Sharded tensor must have shard shape that is a multiple of "
                            "TILE_SIZE!");
                    }
                }
                return ttnn::tilize(
                    tensor,
                    output_memory_config,
                    dtype,
                    use_multicore_tilize,
                    false /* low perf mode */,
                    sub_core_grids);
            }
            throw std::runtime_error("ttnn::to_layout: Unsupported layout!");
        }
        if (layout == ttnn::ROW_MAJOR_LAYOUT) {
            TT_FATAL(
                !dtype.has_value() || dtype.value() == tensor_arg.dtype(),
                "dtype cannot be different from tensor dtype when converting to ROW_MAJOR_LAYOUT on device!");

            if (tensor.is_sharded()) {
                output_memory_config =
                    memory_config.value_or(ttnn::get_memory_config(tensor).value_or(ttnn::DRAM_MEMORY_CONFIG));
            }
            Shape output_tensor_end(SmallVector<uint32_t>(tensor.logical_shape().rank(), 0));
            int logical_rank = tensor.logical_shape().rank();
            for (int index = -1; index >= -logical_rank; --index) {
                output_tensor_end[index] = tensor.logical_shape()[index] - 1;
            }
            tensor = ttnn::untilize_with_unpadding(
                tensor,
                output_tensor_end,
                output_memory_config,
                use_multicore_untilize,
                true /*use_pack_untilize*/,
                sub_core_grids);
            return ttnn::reshape(
                tensor,
                ttnn::Shape{output_shape},
                std::nullopt /*Memory Config*/,
                std::nullopt /*pad value*/,
                TileReshapeMapMode::CACHE,
                sub_core_grids);
        }
        if (layout == ttnn::TILE_LAYOUT) {
            if (tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
                // ttnn::tilize_with_val_padding doesn't support height sharded tensors
                // workaround by applying padding and then tilizing
                SmallVector<std::array<uint32_t, 2>> padding = {
                    {0, 0},
                    {0, 0},
                    {0, padded_output_shape[2] - output_shape[2]},
                    {0, padded_output_shape[3] - output_shape[3]}};
                TT_FATAL(!sub_core_grids.has_value(), "Pad OP does not currently support sub core grid");
                tensor = ttnn::pad(tensor, padding, 0, true, std::nullopt);
                return ttnn::tilize(tensor, output_memory_config, dtype, use_multicore_tilize);
            } else {
                PadValue pad_value_variant;
                if (tensor.dtype() == ttnn::DataType::BFLOAT16 or tensor.dtype() == ttnn::DataType::FLOAT32) {
                    pad_value_variant = 0.0f;
                } else {
                    pad_value_variant = (uint32_t)0;
                }
                tensor = ttnn::tilize_with_val_padding(
                    tensor,
                    Shape(padded_output_shape),
                    pad_value_variant,
                    output_memory_config,
                    dtype,
                    use_multicore_tilize,
                    sub_core_grids);
            }
            if (original_rank == 1) {
                return ttnn::reshape(
                    tensor,
                    original_shape,
                    std::nullopt /*Memory Config*/,
                    std::nullopt /*pad value*/,
                    TileReshapeMapMode::CACHE,
                    sub_core_grids);
            }

            return ttnn::reshape(
                tensor,
                output_shape,
                padded_output_shape,
                std::nullopt, /*Memory Config*/
                std::nullopt, /*Pad Value*/
                TileReshapeMapMode::CACHE,
                sub_core_grids);
        }
```

**File:** ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_device_operation.cpp (L22-43)
```cpp
    const auto& input_tensor_a = tensor_args.input_tensor;
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to tilize need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands to tilize need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.layout() == Layout::ROW_MAJOR, "Can only tilize row major data");

    TT_FATAL(
        input_tensor_a.physical_volume() % tt::constants::TILE_HW == 0,
        "Input tensor physical volume ({}) must be divisible by TILE_HW ({})",
        input_tensor_a.physical_volume(),
        tt::constants::TILE_HW);

    auto width = input_tensor_a.padded_shape()[-1];
    uint32_t stick_s = width;
    TT_FATAL(
        input_tensor_a.dtype() == DataType::BFLOAT16 or input_tensor_a.dtype() == DataType::FLOAT32 or
            input_tensor_a.dtype() == DataType::UINT32 or input_tensor_a.dtype() == DataType::INT32 or
            input_tensor_a.dtype() == DataType::UINT16,
        "data type must be bfloat16, float32, uint32, int32, or uint16");

    uint32_t stick_size = stick_s * input_tensor_a.element_size();  // Assuming bfloat16 dataformat

    TT_FATAL((stick_size % 2) == 0, "Stick size must be divisible by 2");
```

**File:** ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_device_operation.cpp (L112-165)
```cpp
TilizeDeviceOperation::program_factory_t TilizeDeviceOperation::select_program_factory(
    const TilizeDeviceOperation::operation_attributes_t& operation_attributes,
    const TilizeDeviceOperation::tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor;

    bool use_single_core = (operation_attributes.use_low_perf) || (!operation_attributes.use_multicore) ||
                           (operation_attributes.sub_core_grids.has_value() &&
                            (operation_attributes.sub_core_grids.value().num_cores() < 2));
    if (use_single_core) {
        return ttnn::prim::TilizeSingleCoreProgramFactory{};
    }

    if (input_tensor_a.memory_config().is_sharded()) {
        TT_FATAL(
            !operation_attributes.sub_core_grids.has_value(),
            "Sharded tilize does not support sub core grid specification");
        if (input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
            return ttnn::prim::TilizeMultiCoreWidthShardedProgramFactory{};
        }
        return ttnn::prim::TilizeMultiCoreShardedProgramFactory{};
    }
    if (!operation_attributes.enough_space_height) {
        return ttnn::prim::TilizeMultiCoreBlockProgramFactory{};
    }
    auto sub_core_grids = operation_attributes.sub_core_grids;

    uint32_t num_tiles_per_row = input_tensor_a.padded_shape()[-1] / tt::constants::TILE_WIDTH;

    uint32_t num_tiles_per_col = input_tensor_a.padded_shape()[-2] / tt::constants::TILE_HEIGHT;

    int32_t ntiles = input_tensor_a.physical_volume() / tt::constants::TILE_HW;
    uint32_t ntiles_per_block = input_tensor_a.padded_shape()[-1] / tt::constants::TILE_WIDTH;
    uint32_t nblocks = std::ceil(static_cast<float>(ntiles) / ntiles_per_block);

    auto* device = input_tensor_a.device();
    auto grid_size = device->compute_with_storage_grid_size();
    CoreRange default_cores({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    CoreRangeSet default_grid(default_cores);
    CoreRangeSet available_grid = sub_core_grids.has_value() ? sub_core_grids.value() : default_grid;

    size_t grid_area = available_grid.num_cores();
    auto [ncores, nblocks_per_core] = compute_ncores(grid_area, nblocks);
    constexpr uint32_t threshold_row_block = 32;
    if (num_tiles_per_row > threshold_row_block &&
        (num_tiles_per_col > threshold_row_block || num_tiles_per_row > num_tiles_per_col)) {
        uint32_t num_blocks_block = (input_tensor_a.padded_shape()[-1] * input_tensor_a.padded_shape()[-2]) /
                                    (tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH);
        auto ncores_wh = compute_ncores_wh(grid_area, num_blocks_block, num_tiles_per_row, num_tiles_per_col);
        if (ncores < ncores_wh.ncores) {
            return ttnn::prim::TilizeMultiCoreBlockProgramFactory{};
        }
    }
    return ttnn::prim::TilizeMultiCoreInterleavedProgramFactory{};
}
```

**File:** ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/tilize.cpp (L1-26)
```cpp
// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/tilize.h"

void kernel_main() {
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(1);
    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(2);
    constexpr uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(3);
    compute_kernel_hw_startup(cb_id_in0, cb_id_out0);
    tilize_init(cb_id_in0, per_core_block_tile_cnt, cb_id_out0);

    for (uint32_t b = 0; b < per_core_block_cnt; ++b) {
        cb_wait_front(cb_id_in0, per_core_block_tile_cnt);
        cb_reserve_back(cb_id_out0, per_core_block_tile_cnt);

        tilize_block(cb_id_in0, per_core_block_tile_cnt, cb_id_out0);

        cb_push_back(cb_id_out0, per_core_block_tile_cnt);
        cb_pop_front(cb_id_in0, per_core_block_tile_cnt);
    }
}
```

**File:** ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/tilize_wh.cpp (L1-28)
```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/tilize.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
// #include "api/debug/dprint.h"

void kernel_main() {
    const uint32_t block_size_col = get_compile_time_arg_val(0);
    const uint32_t block_size_row = get_compile_time_arg_val(1);
    const uint32_t third_dim = get_compile_time_arg_val(2);

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    tilize_init(tt::CBIndex::c_0, block_size_row, tt::CBIndex::c_16);
    for (uint32_t b = 0; b < block_size_col * third_dim; ++b) {
        cb_wait_front(tt::CBIndex::c_0, block_size_row);
        cb_reserve_back(tt::CBIndex::c_16, block_size_row);

        tilize_block(tt::CBIndex::c_0, block_size_row, tt::CBIndex::c_16);

        cb_push_back(tt::CBIndex::c_16, block_size_row);
        cb_pop_front(tt::CBIndex::c_0, block_size_row);
    }
    tilize_uninit(tt::CBIndex::c_0, tt::CBIndex::c_16);
}
```

**File:** tt_metal/hw/inc/api/compute/tilize.h (L34-46)
```text
ALWI void tilize_init(uint32_t icb, uint32_t block, uint32_t ocb, uint32_t call_line = __builtin_LINE()) {
    state_configure<Operand::SRCA, Operand::PACK>(icb, ocb, call_line);
    UNPACK((llk_unpack_tilize_init(icb, block)));
    MATH((llk_math_eltwise_unary_datacopy_init<
          A2D,
          DST_ACCUM_MODE,
          BroadcastType::NONE,
          false /*is_int_en*/,
          true /*tilize en*/>(icb)));
#ifdef ARCH_BLACKHOLE
    PACK((llk_pack_init<false /*untilize*/, false /*zero output*/, true /*tilize en*/>(ocb)));
#endif
}
```

**File:** ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.cpp (L61-113)
```cpp
ttnn::Tensor ExecuteTilizeWithValPadding::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Shape& output_padded_shape,
    const PadValue pad_value,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DataType> output_dtype,
    bool use_multicore,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    if (input_tensor.layout() == Layout::TILE) {
        return input_tensor;
    }

    // Handle empty tensors - no tiling needed for tensors with no data
    if (input_tensor.physical_volume() == 0) {
        // Create output tensor with same properties
        TensorSpec spec(
            output_padded_shape,
            TensorLayout(
                output_dtype.value_or(input_tensor.dtype()),
                PageConfig(Layout::TILE),
                memory_config.value_or(input_tensor.memory_config())));
        return create_device_tensor(spec, input_tensor.device());
    }

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    uint32_t output_single_tile_size =
        output_dtype.has_value() ? tt::tile_size(tt::tt_metal::datatype_to_dataformat_converter(output_dtype.value()))
                                 : input_single_tile_size;

    uint32_t num_tiles_per_row = output_padded_shape[-1] / tt::constants::TILE_WIDTH;
    uint32_t num_tiles_per_col = output_padded_shape[-2] / tt::constants::TILE_HEIGHT;

    bool enough_space_width =
        is_enough_space(input_tensor, input_single_tile_size, output_single_tile_size, num_tiles_per_col);
    bool enough_space_height =
        is_enough_space(input_tensor, input_single_tile_size, output_single_tile_size, num_tiles_per_row);

    auto base_tilize = [=](const ttnn::Tensor& input_tensor) {
        return ttnn::prim::tilize_with_val_padding(
            input_tensor,
            squeeze_output_shape(output_padded_shape),
            pad_value,
            memory_config.value_or(input_tensor.memory_config()),
            output_dtype.value_or(input_tensor.dtype()),
            use_multicore,
            enough_space_width,
            enough_space_height,
            sub_core_grids);
    };

    return build_ndiml_tilize_val(base_tilize, sub_core_grids)(input_tensor);
}
```

**File:** ttnn/cpp/ttnn/operations/copy/typecast/typecast.hpp (L9-31)
```text
namespace ttnn {

namespace operations::copy {

struct Typecast {
    static Tensor invoke(
        const Tensor& input,
        const DataType& output_dtype,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

    static ttnn::Tensor invoke(
        const Tensor& input_tensor,
        const DataType& tt_input_dtype,
        const DataType& tt_output_dtype,
        const std::optional<MemoryConfig>& memory_config = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
};
}  // namespace operations::copy

constexpr auto typecast = ttnn::register_operation<"ttnn::typecast", ttnn::operations::copy::Typecast>();
```

**File:** ttnn/cpp/ttnn/operations/copy/typecast/typecast.cpp (L34-58)
```cpp
    // Device tensor path
    DataType input_dtype = input_tensor.dtype();
    bool preserve_fp32_precision =
        (input_dtype == DataType::FLOAT32) or
        (output_dtype == DataType::UINT8 and (input_dtype == DataType::BFLOAT16 or input_dtype == DataType::BFLOAT8_B or
                                              input_dtype == DataType::BFLOAT4_B)) or
        (input_dtype == DataType::UINT16 and output_dtype == DataType::UINT8) or
        (input_dtype == DataType::UINT8 and output_dtype != DataType::BFLOAT16);
    bool fp32_dest_acc_en = preserve_fp32_precision or output_dtype == DataType::UINT32 or
                            output_dtype == DataType::INT32 or output_dtype == DataType::FLOAT32 or
                            input_dtype == DataType::UINT32 or input_dtype == DataType::INT32;
    bool bfp8_pack_precise = (output_dtype == DataType::BFLOAT8_B);
    auto output_memory_config = optional_output_tensor.has_value()
                                    ? optional_output_tensor.value().memory_config()
                                    : memory_config.value_or(input_tensor.memory_config());
    return ttnn::prim::typecast(
        input_tensor,
        output_dtype,
        output_memory_config,
        fp32_dest_acc_en,
        preserve_fp32_precision,
        bfp8_pack_precise,
        optional_output_tensor,
        sub_core_grids);
}
```

**File:** tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_typecast.h (L90-98)
```text
    } else if constexpr (in_format == DataFormat::Bfp8_b && out_format == DataFormat::Float16_b) {
        // no SFPU kernel needed, handled by unpacker
    } else if constexpr (in_format == DataFormat::Float16_b && out_format == DataFormat::Bfp8_b) {
        // no SFPU kernel needed, handled by packer
    } else if constexpr (in_format == DataFormat::Bfp8_b && out_format == DataFormat::Float32) {
        // no SFPU kernel needed, handled by unpacker/packer
    } else if constexpr (in_format == DataFormat::Float32 && out_format == DataFormat::Bfp8_b) {
        // no SFPU kernel needed, handled by packer
    } else if constexpr (in_format == DataFormat::Bfp4_b && out_format == DataFormat::UInt16) {
```

**File:** tt_metal/jit_build/data_format.cpp (L217-224)
```cpp
    } else if (fp32_dest_acc_en) {
        if (is_bfp_format(data_format)) {
            if (bfp8_pack_precise) {
                pack_src_format = DataFormat::Float32;
            } else {
                pack_src_format = is_exp_b_format(data_format) ? DataFormat::Bfp8_b : DataFormat::Bfp8;
            }
        } else if (is_exp_b_format(data_format) || (data_format == DataFormat::Float32)) {
```

**File:** tt_metal/jit_build/data_format.cpp (L270-275)
```cpp
        } else if (is_bfp_format(data_format)) {
            if (bfp8_pack_precise) {
                pack_src_format = is_exp_b_format(data_format) ? DataFormat::Float16_b : DataFormat::Float16;
            } else {
                pack_src_format = is_exp_b_format(data_format) ? DataFormat::Bfp8_b : DataFormat::Bfp8;
            }
```

**File:** ttnn/cpp/ttnn/operations/data_movement/untilize/untilize.hpp (L13-25)
```text
struct ExecuteUntilize {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        bool use_multicore = true,
        bool use_pack_untilize = true,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
};

}  // namespace operations::data_movement

constexpr auto untilize =
    ttnn::register_operation<"ttnn::untilize", ttnn::operations::data_movement::ExecuteUntilize>();
```

**File:** ttnn/cpp/ttnn/operations/data_movement/untilize/untilize_nanobind.cpp (L17-59)
```cpp
void bind_untilize(nb::module_& mod) {
    const auto* doc =
        R"doc(
            Changes data layout of input tensor to ROW_MAJOR.

            Input tensor must be on TT accelerator device, in TILE layout, and have BFLOAT16 data type.

            Output tensor will be on TT accelerator device, in ROW_MAJOR layout, and have BFLOAT16 data type.

            Args:
                input_tensor (ttnn.Tensor): the input tensor.

            Keyword Args:

                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
                use_multicore (bool, optional): Whether to use multicore. Defaults to `True`.
                use_pack_untilize (bool, optional): Whether to use pack untilize. Defaults to `True`.
                sub_core_grids (ttnn.CoreRangeSet, optional): Sub core grids. Defaults to `None`.

            Returns:
                List of ttnn.Tensor: the output tensor.
        )doc";

    using OperationType = decltype(ttnn::untilize);
    ttnn::bind_registered_operation(
        mod,
        ttnn::untilize,
        doc,
        ttnn::nanobind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               const std::optional<MemoryConfig>& memory_config,
               bool use_multicore,
               bool use_pack_untilize,
               const std::optional<CoreRangeSet>&& sub_core_grids) {
                return self(input_tensor, memory_config, use_multicore, use_pack_untilize, sub_core_grids);
            },
            nb::arg("input_tensor"),
            nb::kw_only(),
            nb::arg("memory_config") = nb::none(),
            nb::arg("use_multicore") = true,
            nb::arg("use_pack_untilize") = true,
            nb::arg("sub_core_grids") = nb::none()});
```

**File:** ttnn/cpp/ttnn/operations/data_movement/untilize/device/untilize_device_operation.cpp (L271-285)
```cpp
UntilizeDeviceOperation::spec_return_value_t UntilizeDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::constants;
    const auto& input_tensor = tensor_args.input;
    DataType output_dtype = input_tensor.dtype() == DataType::BFLOAT8_B ? DataType::BFLOAT16 : input_tensor.dtype();

    return {TensorSpec(
        input_tensor.logical_shape(),
        TensorLayout::fromPaddedShape(
            output_dtype,
            PageConfig(Layout::ROW_MAJOR),
            operation_attributes.output_mem_config,
            input_tensor.logical_shape(),
            input_tensor.padded_shape()))};
}
```

**File:** ttnn/cpp/ttnn/operations/data_movement/untilize/device/untilize_device_operation.cpp (L292-383)
```cpp
UntilizeDeviceOperation::program_factory_t UntilizeDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input;
    const auto& output_tensor = operation_attributes.output_mem_config;

    bool input_is_sharded = input_tensor_a.is_sharded();
    bool output_is_sharded = output_tensor.is_sharded();

    BufferType input_buffer_type = input_tensor_a.memory_config().buffer_type();
    BufferType output_buffer_type = output_tensor.buffer_type();

    TensorMemoryLayout input_memory_layout = input_tensor_a.memory_config().memory_layout();
    TensorMemoryLayout output_memory_layout = output_tensor.memory_layout();

    if (!operation_attributes.use_multicore) {
        // Single core implementation
        return UntilizeSingleCoreProgramFactory{};
    }
    if (operation_attributes.sub_core_grids.has_value()) {
        // If sub_core_grids parameter is provided, use custom sub_core_grid implementation instead
        // of the standard multicore implementation or the block multicore implementation.
        // Note that this implementation does not support sharding, which is enforced in validate().
        return UntilizeMultiCoreSubCoreGridsProgramFactory{};
    }
    if (!operation_attributes.enough_space_height && !input_is_sharded && !output_is_sharded) {
        // Optimized special case implementation, only supported when neither input or output is sharded
        return UntilizeMultiCoreBlockProgramFactory{};
    }
    if (input_is_sharded && output_is_sharded && input_buffer_type == BufferType::L1 &&
        output_buffer_type == BufferType::L1 && input_memory_layout == output_memory_layout) {
        // Optimized special case implementation for when both input and output are sharded, both are located in L1,
        // have identical memory layouts (i.e. height->height, width->width, block->block), and have identical shard
        // specs
        bool identical_shard_specs = false;
        identical_shard_specs |= input_tensor_a.shard_spec().has_value() && output_tensor.shard_spec().has_value() &&
                                 input_tensor_a.shard_spec().value() == output_tensor.shard_spec().value();
        if (identical_shard_specs) {
            return UntilizeMultiCoreInputAndOutputShardTypeAndShardSpecIdenticalProgramFactory{};
        }
        identical_shard_specs |= input_tensor_a.nd_shard_spec().has_value() &&
                                 output_tensor.nd_shard_spec().has_value() &&
                                 input_tensor_a.nd_shard_spec().value() == output_tensor.nd_shard_spec().value();

        if (identical_shard_specs) {
            return UntilizeMultiCoreInputAndOutputNDShardTypeAndShardSpecIdenticalProgramFactory{};
        }
    }

    uint32_t tensor_width = input_tensor_a.padded_shape()[-1];
    uint32_t tensor_height = input_tensor_a.physical_volume() / tensor_width;

    const auto& tile_shape = input_tensor_a.tensor_spec().tile().get_tile_shape();
    uint32_t tile_height = tile_shape[0];
    uint32_t tile_width = tile_shape[1];

    uint32_t num_tiles_per_row = tensor_width / tile_width;
    uint32_t num_tiles_per_col = tensor_height / tile_height;

    auto grid_size = input_tensor_a.device()->compute_with_storage_grid_size();

    size_t grid_area = grid_size.x * grid_size.y;
    auto [num_compute_cores, nblocks_per_core] = compute_ncores(grid_area, num_tiles_per_col);

    constexpr uint32_t threshold_row_block = 32;
    if (!input_is_sharded and !output_is_sharded) {
        if (num_tiles_per_row > threshold_row_block and
            (num_tiles_per_col > threshold_row_block or num_tiles_per_row > num_tiles_per_col)) {
            uint32_t num_blocks_block = (input_tensor_a.padded_shape()[-1] * input_tensor_a.padded_shape()[-2]) /
                                        (tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH);
            auto ncores_wh = compute_ncores_wh(grid_area, num_blocks_block, num_tiles_per_row, num_tiles_per_col);
            if (num_compute_cores < ncores_wh.ncores) {
                return UntilizeMultiCoreBlockProgramFactory{};
            }
        }
    }
    // TODO : currently multi_core parallelization on column only works for single tile height tensors.
    // Need to debug this to work on wide tensors that are higher than a single tile
    auto pf_option = ttnn::operations::data_movement::get_pf_type(output_is_sharded, input_tensor_a);
    if (pf_option == 0) {
        return UntilizeMultiCoreParallelizeColumnProgramFactory{};
    }
    if (pf_option == 1) {
        return UntilizeSingleCoreProgramFactory{};
    }

    if (not input_is_sharded or input_tensor_a.shard_spec().has_value()) {
        // default multi core implementation, non ND-sharded input
        return UntilizeMultiCoreProgramFactory{};
    }
    // Default ND shard multi core implementation
    return UntilizeMultiCoreNDShardInputProgramFactory{};
}
```

**File:** ttnn/cpp/ttnn/operations/data_movement/untilize/device/untilize_device_operation.cpp (L398-406)
```cpp
    const int max_tiles_per_row = 8;
    const int latency_untilize = 390;      // measured latency for untilize_block
    const int latency_pack_untilize = 80;  // measured latency for pack_untilize_block
    if (std::ceil(static_cast<float>(input_tensor.padded_shape()[-1]) / static_cast<float>(tile_width)) <=
        max_tiles_per_row) {
        compute_cycles = num_tiles * latency_pack_untilize;
    } else {
        compute_cycles = num_tiles * latency_untilize;
    }
```

**File:** ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize.cpp (L1-38)
```cpp
// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/untilize.h"
#include "api/compute/pack_untilize.h"
#include "common.cpp"

void kernel_main() {
#ifdef DST_ACCUM_MODE
    constexpr uint32_t max_bct = 4;
#else
    constexpr uint32_t max_bct = 8;
#endif
    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    constexpr uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);
    constexpr uint32_t src_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(3);

    // Compute optimal num_blocks_per_col and block_ct_dim
    constexpr uint32_t num_blocks_per_col = compute_num_blocks_per_column(per_core_block_tile_cnt, max_bct);
    constexpr uint32_t block_ct_dim = per_core_block_tile_cnt / num_blocks_per_col;
    constexpr uint32_t full_ct_dim = per_core_block_tile_cnt;
    compute_kernel_hw_startup(src_cb_id, out_cb_id);
    pack_untilize_init<block_ct_dim, full_ct_dim>(src_cb_id, out_cb_id);

    for (uint32_t r = 0; r < per_core_block_cnt; ++r) {
        cb_reserve_back(out_cb_id, full_ct_dim);
        for (uint32_t b = 0; b < num_blocks_per_col; ++b) {
            cb_wait_front(src_cb_id, block_ct_dim);
            pack_untilize_block<block_ct_dim, full_ct_dim>(src_cb_id, 1, out_cb_id, b);
            cb_pop_front(src_cb_id, block_ct_dim);
        }
        cb_push_back(out_cb_id, full_ct_dim);
    }
    pack_untilize_uninit(out_cb_id);
```

**File:** ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize_wh.cpp (L1-38)
```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/untilize.h"
#include "api/compute/pack_untilize.h"
#include "api/debug/dprint.h"
#include "common.cpp"

void kernel_main() {
#ifdef DST_ACCUM_MODE
    constexpr uint32_t max_bct = 4;
#else
    constexpr uint32_t max_bct = 8;
#endif
    const uint32_t block_size_col = get_compile_time_arg_val(0);
    const uint32_t block_size_row = get_compile_time_arg_val(1);
    const uint32_t third_dim = get_compile_time_arg_val(2);

    // Compute optimal num_blocks_per_col and block_ct_dim
    constexpr uint32_t num_blocks_per_col = compute_num_blocks_per_column(block_size_row, max_bct);
    constexpr uint32_t block_ct_dim = block_size_row / num_blocks_per_col;
    constexpr uint32_t full_ct_dim = block_size_row;

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    pack_untilize_init<block_ct_dim, full_ct_dim>(tt::CBIndex::c_0, tt::CBIndex::c_16);

    for (uint32_t b = 0; b < block_size_col * third_dim; ++b) {
        cb_reserve_back(tt::CBIndex::c_16, full_ct_dim);
        for (uint32_t b = 0; b < num_blocks_per_col; ++b) {
            cb_wait_front(tt::CBIndex::c_0, block_ct_dim);
            pack_untilize_block<block_ct_dim, full_ct_dim>(tt::CBIndex::c_0, 1, tt::CBIndex::c_16, b);
            cb_pop_front(tt::CBIndex::c_0, block_ct_dim);
        }
        cb_push_back(tt::CBIndex::c_16, full_ct_dim);
    }
    pack_untilize_uninit(tt::CBIndex::c_16);
}
```

**File:** ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp (L1-26)
```cpp
// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/untilize.h"

void kernel_main() {
    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    constexpr uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);
    constexpr uint32_t src_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(3);

    compute_kernel_hw_startup(src_cb_id, out_cb_id);
    untilize_init(src_cb_id);

    for (uint32_t b = 0; b < per_core_block_cnt; ++b) {
        cb_wait_front(src_cb_id, per_core_block_tile_cnt);
        cb_reserve_back(out_cb_id, per_core_block_tile_cnt);

        untilize_block(src_cb_id, per_core_block_tile_cnt, out_cb_id);

        cb_push_back(out_cb_id, per_core_block_tile_cnt);
        cb_pop_front(src_cb_id, per_core_block_tile_cnt);
    }
```

**File:** tt_metal/impl/data_format/bfloat8.cpp (L23-27)
```cpp
template <typename T>
std::vector<uint32_t> pack_as_bfp8_tiles(
    tt::stl::Span<const T> data, bool row_major_input, bool is_exp_a, const std::optional<tt::tt_metal::Tile>& tile) {
    return pack_as_bfp_tiles<tt::DataFormat::Bfp8_b>(data, row_major_input, is_exp_a, tile);
}
```

**File:** ttnn/core/tensor/py_to_tt_tensor.cpp (L22-44)
```cpp
bool can_exec_ops_on_device(DataType type) {
    switch (type) {
        case DataType::BFLOAT16:
            // https://github.com/tenstorrent/tt-metal/issues/31406 (NaN values are not preserved and replaced with inf)
        case DataType::FLOAT32:
            // https://github.com/tenstorrent/tt-metal/issues/23405 (layout precision loss)
            // https://github.com/tenstorrent/tt-metal/issues/30147 (typecast rounding error)
        case DataType::UINT32:
        case DataType::INT32:
            // https://github.com/tenstorrent/tt-metal/issues/23407 (to_layout(RM) is not working for uint32/int32)
        case DataType::UINT16:
            // Tilize doesn't support uint16.
        case DataType::UINT8:
            // https://github.com/tenstorrent/tt-metal/issues/21682 (typecast doesn't support uint8)
        case DataType::BFLOAT4_B:
        case DataType::BFLOAT8_B:
            // https://github.com/tenstorrent/tt-metal/issues/35048
            // Conversion from bfloat16 to bfloat4_b or bfloat8_b loses precision.
            // The test triggering this bug is test_matmul.py::test_tiny_tiles_bfloat
            return false;
        default: return true;
    }
};
```

**File:** ttnn/core/tensor/py_to_tt_tensor.cpp (L175-192)
```cpp
    const DataType mapped_dst_type =
        (dst_dtype == DataType::BFLOAT4_B or dst_dtype == DataType::BFLOAT8_B) ? DataType::FLOAT32 : dst_dtype;

    if (!is_torch_dtype_matches_ttnn(src_dtype)) {
        return mapped_dst_type;
    }

    if (is_sharded && get_datatype_tile_size(dst_dtype) != get_datatype_tile_size(to_ttnn_dtype(src_dtype))) {
        // Sharded typecast does not support conversion between tensors with types of different tile size:
        // See explicit assertion in the `TypecastShardedProgramFactory::create` method implementation.
        return mapped_dst_type;
    }

    // TODO: Perform type conversion on the Python side for now due to performance considerations.
    // Device-side type conversion is disabled because of issues with typecast/to_layout.
    // Re-enable device-side typecasting once these issues are fixed.
    return mapped_dst_type;
    // return to_ttnn_dtype(src_dtype);  // borrow pytensor by default.
```

**File:** ttnn/core/tensor/py_to_tt_tensor.cpp (L232-268)
```cpp
    if (col_tilize) {
        // Transpose the last two dims of the float32 host buffer by creating a new
        // buffer and replacing host_buffer, so that BFP exponent grouping happens
        // along columns instead of rows.
        auto rank = tensor_shape.rank();
        TT_FATAL(rank >= 2, "col_tilize requires tensor rank >= 2, got {}", rank);
        TT_FATAL(
            dst_dtype == DataType::BFLOAT8_B || dst_dtype == DataType::BFLOAT4_B,
            "col_tilize requires BFP dtype (BFLOAT8_B or BFLOAT4_B)");

        const auto K = tensor_shape[-2];
        const auto N = tensor_shape[-1];
        size_t batch_size = 1;
        for (size_t i = 0; i < rank - 2; ++i) {
            batch_size *= tensor_shape[i];
        }

        const float* src = host_buffer.view_as<float>().data();
        std::vector<float> transposed(batch_size * K * N);
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t i = 0; i < static_cast<size_t>(N); ++i) {
                for (size_t j = 0; j < static_cast<size_t>(K); ++j) {
                    transposed[(b * N * K) + (i * K) + j] = src[(b * K * N) + (j * N) + i];
                }
            }
        }
        host_buffer = HostBuffer(std::move(transposed));

        // Build transposed shape: swap last two dims
        std::vector<uint32_t> new_dims;
        new_dims.reserve(rank);
        for (size_t i = 0; i < rank - 2; ++i) {
            new_dims.push_back(tensor_shape[i]);
        }
        new_dims.push_back(N);
        new_dims.push_back(K);
        effective_shape = ttnn::Shape(tt::stl::Span<const uint32_t>(new_dims.data(), new_dims.size()));
```

**File:** ttnn/core/tensor/py_to_tt_tensor.cpp (L290-298)
```cpp
    if (device) {
        output = output.to_device(device.value(), memory_config, cq_id);
        if (output.dtype() != dst_dtype) {
            // Need to perform final data conversion on device, typecast requires TILE layout.
            set_layout(Layout::TILE);
            output = ttnn::typecast(output, dst_dtype);
        }

        set_layout(layout);
```
