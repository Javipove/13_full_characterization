// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <stdint.h>

void kernel_main() {
  // DRAM buffer address using InterleavedAddrGen
  uint32_t in0_tensor_addr = get_arg_val<uint32_t>(0);
  uint32_t in0_tensor_start_tile_id = get_arg_val<uint32_t>(1);
  uint32_t in0_tensor_stride_w = get_arg_val<uint32_t>(2);
  uint32_t in0_tensor_stride_h = get_arg_val<uint32_t>(3);
  uint32_t in0_tensor_next_block_stride = get_arg_val<uint32_t>(4);

  // in0 block args
  uint32_t in0_block_w = get_arg_val<uint32_t>(5);
  uint32_t in0_block_h = get_arg_val<uint32_t>(6);
  uint32_t in0_block_num_tiles = get_arg_val<uint32_t>(7);

  // in0/in1 common args
  uint32_t num_blocks = get_arg_val<uint32_t>(8);

  uint32_t noc_x = get_arg_val<uint32_t>(9);
  uint32_t noc_y = get_arg_val<uint32_t>(10);
  uint32_t last_block_h = get_arg_val<uint32_t>(11);

  uint32_t num_blocks_h_dim = get_arg_val<uint32_t>(12);
  uint32_t num_blocks_w_dim = get_arg_val<uint32_t>(13);
  uint32_t in0_h_dim_stride = get_arg_val<uint32_t>(14); // out_block_h * Kt

  constexpr uint32_t cb_id_in0 = 0;
  constexpr uint32_t cb_id_in2 = 2; // Zeros buffer

  // Use Interleaved DRAM Reading
  const uint32_t in0_single_tile_size_bytes = get_tile_size(cb_id_in0);
  const DataFormat in0_data_format = get_dataformat(cb_id_in0);

  const InterleavedAddrGenFast<true> s0 = {.bank_base_address = in0_tensor_addr,
                                           .page_size =
                                               in0_single_tile_size_bytes,
                                           .data_format = in0_data_format};

  // Fill tile with zeros (if needed for padding)
  cb_reserve_back(cb_id_in2, 1);
  uint64_t l1_zeros_addr_in2_noc = get_noc_addr(get_write_ptr(cb_id_in2));

  uint32_t l1_write_addr_in0;

  for (uint32_t bh = 0; bh < num_blocks_h_dim; bh++) {
    for (uint32_t bw = 0; bw < num_blocks_w_dim; bw++) {
      uint32_t in0_tensor_current_block_start_tile_id =
          in0_tensor_start_tile_id;
      for (uint32_t block = 0; block < num_blocks; block++) {
        cb_reserve_back(cb_id_in0, in0_block_num_tiles);
        l1_write_addr_in0 = get_write_ptr(cb_id_in0);

        uint32_t in0_tensor_row_start_tile_id =
            in0_tensor_current_block_start_tile_id;
        for (uint32_t h = 0; h < in0_block_h; h++) {
          uint32_t in0_tensor_tile_id = in0_tensor_row_start_tile_id;

          for (uint32_t w = 0; w < in0_block_w; w++) {
            if (h < last_block_h) {
              noc_async_read_tile(in0_tensor_tile_id, s0, l1_write_addr_in0);
            } else {
              noc_async_read(l1_zeros_addr_in2_noc, l1_write_addr_in0,
                             in0_single_tile_size_bytes);
            }
            l1_write_addr_in0 += in0_single_tile_size_bytes;
            in0_tensor_tile_id += in0_tensor_stride_w;
          }
          in0_tensor_row_start_tile_id += in0_tensor_stride_h;
        }

        in0_tensor_current_block_start_tile_id += in0_tensor_next_block_stride;

        noc_async_read_barrier();
        cb_push_back(cb_id_in0, in0_block_num_tiles);
      }
    } // bw — A is re-read for each N-block
    in0_tensor_start_tile_id += in0_h_dim_stride; // advance to next M-block
  } // bh
}
