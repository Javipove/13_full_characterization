// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <stdint.h>

void kernel_main() {
  // DRAM buffer address usage InterleavedAddrGen
  uint32_t in1_tensor_addr = get_arg_val<uint32_t>(0);
  uint32_t in1_tensor_start_tile_id = get_arg_val<uint32_t>(1);
  uint32_t in1_tensor_stride_w = get_arg_val<uint32_t>(2);
  uint32_t in1_tensor_stride_h = get_arg_val<uint32_t>(3);
  uint32_t in1_tensor_next_block_stride = get_arg_val<uint32_t>(4);

  // in1 block args
  uint32_t in1_block_w = get_arg_val<uint32_t>(5);
  uint32_t in1_block_h = get_arg_val<uint32_t>(6);
  uint32_t in1_block_num_tiles = get_arg_val<uint32_t>(7);

  // in0/in1 common args
  uint32_t num_blocks = get_arg_val<uint32_t>(8);
  uint32_t in2_cb_addr = get_arg_val<uint32_t>(9);

  uint32_t noc_x = get_arg_val<uint32_t>(10);
  uint32_t noc_y = get_arg_val<uint32_t>(11);

  // out tensor args
  uint32_t out_tensor_addr = get_arg_val<uint32_t>(12);
  uint32_t out_tensor_start_tile_id = get_arg_val<uint32_t>(13);
  uint32_t out_tensor_stride_w = get_arg_val<uint32_t>(14);
  uint32_t out_tensor_stride_h = get_arg_val<uint32_t>(15);
  uint32_t out_tensor_next_subblock_stride_w = get_arg_val<uint32_t>(16);
  uint32_t out_tensor_next_subblock_stride_h = get_arg_val<uint32_t>(17);

  // out subblock args
  uint32_t out_subblock_w = get_arg_val<uint32_t>(18);
  uint32_t out_subblock_h = get_arg_val<uint32_t>(19);
  uint32_t out_subblock_tile_count = get_arg_val<uint32_t>(20);
  uint32_t out_num_subblocks_w = get_arg_val<uint32_t>(21);
  uint32_t out_num_subblocks_h = get_arg_val<uint32_t>(22);

  // padding args (in1)
  uint32_t last_block_w = get_arg_val<uint32_t>(23);

  // padding args (writer)
  uint32_t out_num_nonzero_subblocks_h = get_arg_val<uint32_t>(24);
  uint32_t out_last_subblock_h = get_arg_val<uint32_t>(25);
  uint32_t padded_block_tiles_h_skip = get_arg_val<uint32_t>(26);
  uint32_t out_num_nonzero_subblocks_w = get_arg_val<uint32_t>(27);
  uint32_t out_last_subblock_w = get_arg_val<uint32_t>(28);
  uint32_t padded_subblock_tiles_addr_skip = get_arg_val<uint32_t>(29);
  uint32_t padded_block_tiles_w_skip = get_arg_val<uint32_t>(30);

  constexpr uint32_t cb_id_in1 = 1;
  constexpr uint32_t cb_id_in2 = 2; // Zeros
  constexpr uint32_t cb_id_out0 = 16;

  // Use Interleaved DRAM Reading/Writing
  const uint32_t in1_single_tile_size_bytes = get_tile_size(cb_id_in1);
  const DataFormat in1_data_format = get_dataformat(cb_id_in1);
  const InterleavedAddrGenFast<true> s1 = {.bank_base_address = in1_tensor_addr,
                                           .page_size =
                                               in1_single_tile_size_bytes,
                                           .data_format = in1_data_format};

  const uint32_t out_single_tile_size_bytes = get_tile_size(cb_id_out0);
  const DataFormat out_data_format = get_dataformat(cb_id_out0);
  const InterleavedAddrGenFast<true> s_out = {
      .bank_base_address = out_tensor_addr,
      .page_size = out_single_tile_size_bytes,
      .data_format = out_data_format};

  // Fill tile with zeros
  cb_reserve_back(cb_id_in2, 1);
  uint64_t l1_zeros_addr_in2_noc = get_noc_addr(get_write_ptr(cb_id_in2));

  // in1 reader
  uint32_t l1_write_addr_in1;
  uint32_t in1_tensor_current_block_start_tile_id = in1_tensor_start_tile_id;
  for (uint32_t block = 0; block < num_blocks; block++) {
    cb_reserve_back(cb_id_in1, in1_block_num_tiles);
    l1_write_addr_in1 = get_write_ptr(cb_id_in1);

    uint32_t in1_tensor_row_start_tile_id =
        in1_tensor_current_block_start_tile_id;
    for (uint32_t h = 0; h < in1_block_h; h++) {
      uint32_t in1_tensor_tile_id = in1_tensor_row_start_tile_id;
      for (uint32_t w = 0; w < in1_block_w; w++) {
        if (w < last_block_w) {
          noc_async_read_tile(in1_tensor_tile_id, s1, l1_write_addr_in1);
        } else {
          noc_async_read(l1_zeros_addr_in2_noc, l1_write_addr_in1,
                         in1_single_tile_size_bytes);
        }
        l1_write_addr_in1 += in1_single_tile_size_bytes;
        in1_tensor_tile_id += in1_tensor_stride_w;
      }
      in1_tensor_row_start_tile_id += in1_tensor_stride_h;
    }
    // Critical Fix: Update block stride for DRAM
    in1_tensor_current_block_start_tile_id += in1_tensor_next_block_stride;

    noc_async_read_barrier();
    cb_push_back(cb_id_in1, in1_block_num_tiles);
  }

  // writer
  uint32_t out_tensor_sbh_start_tile_id = out_tensor_start_tile_id;
  for (uint32_t sbh = 0; sbh < out_num_nonzero_subblocks_h; sbh++) {
    uint32_t out_tensor_sbw_start_tile_id = out_tensor_sbh_start_tile_id;
    for (uint32_t sbw = 0; sbw < out_num_nonzero_subblocks_w; sbw++) {
      uint32_t out_tensor_sb_row_start_tile_id = out_tensor_sbw_start_tile_id;

      uint32_t out_subblock_h_ = out_subblock_h;
      uint32_t out_subblock_w_ = out_subblock_w;
      uint32_t subblock_tiles_addr_skip = 0;
      if (sbh == out_num_nonzero_subblocks_h - 1) {
        out_subblock_h_ = out_last_subblock_h;
      }
      if (sbw == out_num_nonzero_subblocks_w - 1) {
        out_subblock_w_ = out_last_subblock_w;
        subblock_tiles_addr_skip = padded_subblock_tiles_addr_skip;
      }

      cb_wait_front(cb_id_out0, out_subblock_tile_count);
      uint32_t l1_read_addr = get_read_ptr(cb_id_out0);

      for (uint32_t h = 0; h < out_subblock_h_; h++) {
        uint32_t out_tensor_tile_id = out_tensor_sb_row_start_tile_id;
        for (uint32_t w = 0; w < out_subblock_w_; w++) {
          noc_async_write_tile(out_tensor_tile_id, s_out, l1_read_addr);
          l1_read_addr += out_single_tile_size_bytes;
          out_tensor_tile_id += out_tensor_stride_w;
        }
        // Skip padded tiles in subblock along row
        l1_read_addr += subblock_tiles_addr_skip;
        out_tensor_sb_row_start_tile_id += out_tensor_stride_h;
      }

      noc_async_write_barrier();
      cb_pop_front(cb_id_out0, out_subblock_tile_count);
      out_tensor_sbw_start_tile_id += out_tensor_next_subblock_stride_w;
    }
    // Pop fully padded subblocks along the row
    cb_wait_front(cb_id_out0, padded_block_tiles_w_skip);
    cb_pop_front(cb_id_out0, padded_block_tiles_w_skip);
    out_tensor_sbh_start_tile_id += out_tensor_next_subblock_stride_h;
  }
  // Pop row(s) of fully padded subblocks
  cb_wait_front(cb_id_out0, padded_block_tiles_h_skip);
  cb_pop_front(cb_id_out0, padded_block_tiles_h_skip);
}
