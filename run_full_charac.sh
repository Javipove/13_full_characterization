#!/bin/bash

# Default value for runtime arguments
NUM_RT_ARGS=255
CLEAN_MODE=0

# Search for --num-rt-args in arguments to generate correct kernels
args=("$@")
for ((i=0; i < ${#args[@]}; i++)); do
    if [[ "${args[i]}" == "--num-rt-args" ]]; then
        NUM_RT_ARGS="${args[i+1]}"
    fi
    if [[ "${args[i]}" == "--clean-mode" ]]; then
        CLEAN_MODE="${args[i+1]}"
    fi
done

if [[ "$CLEAN_MODE" == "1" ]]; then
    # Resolve cache directory: TT_METAL_CACHE -> $HOME/.cache -> /tmp
    if [[ -n "${TT_METAL_CACHE}" ]]; then
        CACHE_DIR="${TT_METAL_CACHE}"
    elif [[ -n "${HOME}" ]]; then
        CACHE_DIR="${HOME}/.cache/tt-metal-cache"
    else
        CACHE_DIR="/tmp/tt-metal-cache"
    fi

    echo "Cleaning disk cache at $CACHE_DIR..."
    rm -rf "$CACHE_DIR"
    mkdir -p "$CACHE_DIR"
    # Generate a random seed to force source code change and recompilation
    RANDOM_SEED="// Build Seed: $(date +%s%N)"
else
    RANDOM_SEED="// Build Seed: Default"
fi

echo "Generating kernels with $NUM_RT_ARGS runtime arguments..."

KERNEL_DIR="${TT_METAL_HOME}/tests/tt_metal/tt_metal/perf_microbenchmark/13_full_charac/kernels"
KERNEL_COMMON_DIR="${TT_METAL_HOME}/tests/tt_metal/tt_metal/perf_microbenchmark/13_full_charac/kernels_common"

mkdir -p $KERNEL_DIR

# Copy common kernels
cp $KERNEL_COMMON_DIR/*.cpp $KERNEL_DIR/

# Generate empty_reader.cpp
cat <<EOF > $KERNEL_DIR/empty_reader.cpp
// Auto-generated empty_reader.cpp
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
$RANDOM_SEED

#include <stdint.h>

#include "dataflow_api.h"

void kernel_main() {
    uint32_t compile_arg0 = get_compile_time_arg_val(0);
EOF

for ((i=0; i < NUM_RT_ARGS; i++)); do
    echo "    uint32_t runtime_arg$i = get_arg_val<uint32_t>($i);" >> $KERNEL_DIR/empty_reader.cpp
done

echo "}" >> $KERNEL_DIR/empty_reader.cpp

# Generate empty_writer.cpp
cat <<EOF > $KERNEL_DIR/empty_writer.cpp
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
$RANDOM_SEED

#include "dataflow_api.h"

void kernel_main() {
    uint32_t compile_arg0 = get_compile_time_arg_val(0);
EOF

for ((i=0; i < NUM_RT_ARGS; i++)); do
    echo "    uint32_t runtime_arg$i = get_arg_val<uint32_t>($i);" >> $KERNEL_DIR/empty_writer.cpp
done

echo "}" >> $KERNEL_DIR/empty_writer.cpp

# Execute the benchmark with all passed arguments
# Usage example: ./run_full_charac.sh ./build/test/test_full_charac --num-rt-args 512 ...
echo "Executing: $@"
"$@"
