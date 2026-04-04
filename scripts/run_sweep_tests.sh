#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

TT_METAL_HOME_DEFAULT="/scratch/javier/tt-metal"
VENV_ACTIVATE_DEFAULT="${TT_METAL_HOME_DEFAULT}/venv_tt_javi/bin/activate"

if [[ -f "$VENV_ACTIVATE_DEFAULT" ]]; then
    # shellcheck disable=SC1090
    source "$VENV_ACTIVATE_DEFAULT"
else
    echo "Warning: venv activate script not found at $VENV_ACTIVATE_DEFAULT"
fi

export TT_METAL_HOME="${TT_METAL_HOME:-$TT_METAL_HOME_DEFAULT}"
export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-$TT_METAL_HOME}"
unset TT_METAL_SLOW_DISPATCH_MODE

if [[ ! -x "./run_full_charac.sh" ]]; then
    echo "Error: ./run_full_charac.sh not found or not executable"
    exit 1
fi

if [[ ! -x "./build/test_full_charac" ]]; then
    echo "Error: ./build/test_full_charac not found. Build the project first."
    exit 1
fi

if ! command -v tt-smi >/dev/null 2>&1; then
    echo "Error: tt-smi not found in PATH"
    exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="logs/sweep_${STAMP}"
mkdir -p "$OUT_DIR"
SUMMARY_TSV="$OUT_DIR/summary.tsv"

printf "test_name\tstatus\texit_code\tretry\tlog_file\n" > "$SUMMARY_TSV"

BAR_REGEX="BAR0 UC mapping failed|BAR allocation|BAR0.*mapping|BAR.*allocation"

TEST_NAMES=(
    "test0_empty"
    "test1_dram_cpu_cpu"
    "test1_dram_device_device"
    "test3_host_pipeline_compute"
    "test4_host_pipeline_empty"
    "test5_async_batch"
    "test6_trace_replay"
)

TEST_COMMANDS=(
    "./run_full_charac.sh ./build/test_full_charac --test 0 --num-iters 3 --num-rt-args 4 --cpu 2 --cpu-range 8 --x_size 7 --y_size 7 --m 4096 --n 4096 --k 4096 --bypass-check"
    "./run_full_charac.sh ./build/test_full_charac --test 1 --dram --pack-tile cpu --unpack-tile cpu --num-iters 3 --cpu 2 --cpu-range 8 --x_size 7 --y_size 7 --m 4096 --n 4096 --k 4096 --bypass-check"
    "./run_full_charac.sh ./build/test_full_charac --test 1 --dram --pack-tile device --unpack-tile device --num-iters 3 --cpu 2 --cpu-range 8 --x_size 7 --y_size 7 --m 4096 --n 4096 --k 4096 --bypass-check"
    "./run_full_charac.sh ./build/test_full_charac --test 3 --dram --num-iters 3 --cpu 2 --cpu-range 8 --x_size 7 --y_size 7 --m 4096 --n 4096 --k 4096 --bypass-check"
    "./run_full_charac.sh ./build/test_full_charac --test 4 --dram --num-iters 3 --cpu 2 --cpu-range 8 --x_size 7 --y_size 7 --m 4096 --n 4096 --k 4096 --bypass-check"
    "./run_full_charac.sh ./build/test_full_charac --test 5 --dram --num-iters 10 --cpu 2 --cpu-range 8 --x_size 7 --y_size 7 --m 4096 --n 4096 --k 4096 --bypass-check"
    "./run_full_charac.sh ./build/test_full_charac --test 6 --dram --num-iters 10 --num-rt-args 1 --cpu 2 --cpu-range 8 --x_size 7 --y_size 7 --m 4096 --n 4096 --k 4096 --bypass-check"
)

run_case() {
    local name="$1"
    local cmd="$2"
    local log="$OUT_DIR/${name}.log"
    local retry_log="$OUT_DIR/${name}.retry.log"
    local rc=0
    local retry="no"
    local status=""

    echo
    echo "=== Running $name ==="
    echo "Command: $cmd"

    if [[ $DRY_RUN -eq 1 ]]; then
        status="SKIPPED_DRY_RUN"
        printf "%s\t%s\t%s\t%s\t%s\n" "$name" "$status" "0" "$retry" "$log" >> "$SUMMARY_TSV"
        return
    fi

    tt-smi -r 0 > "$OUT_DIR/${name}.reset.log" 2>&1 || true

    set -o pipefail
    bash -lc "$cmd" |& tee "$log"
    rc=${PIPESTATUS[0]}
    set +o pipefail

    if [[ $rc -eq 0 ]]; then
        status="PASS"
    else
        if grep -Eiq "$BAR_REGEX" "$log"; then
            retry="yes"
            echo "Detected BAR-related failure in $name. Retrying once after reset..."
            tt-smi -r 0 > "$OUT_DIR/${name}.reset_retry.log" 2>&1 || true

            set -o pipefail
            bash -lc "$cmd" |& tee "$retry_log"
            rc=${PIPESTATUS[0]}
            set +o pipefail

            if [[ $rc -eq 0 ]]; then
                status="PASS_AFTER_BAR_RETRY"
                log="$retry_log"
            else
                if grep -Eiq "$BAR_REGEX" "$retry_log"; then
                    status="FAILED_BY_DRIVER"
                    log="$retry_log"
                else
                    status="FAIL"
                    log="$retry_log"
                fi
            fi
        else
            status="FAIL"
        fi
    fi

    printf "%s\t%s\t%s\t%s\t%s\n" "$name" "$status" "$rc" "$retry" "$log" >> "$SUMMARY_TSV"
}

for idx in "${!TEST_NAMES[@]}"; do
    run_case "${TEST_NAMES[$idx]}" "${TEST_COMMANDS[$idx]}"
done

echo
echo "Sweep complete. Summary file: $SUMMARY_TSV"

PASS_COUNT=$(awk -F'\t' 'NR>1 && ($2=="PASS" || $2=="PASS_AFTER_BAR_RETRY") {c++} END {print c+0}' "$SUMMARY_TSV")
FAIL_COUNT=$(awk -F'\t' 'NR>1 && ($2=="FAIL" || $2=="FAILED_BY_DRIVER") {c++} END {print c+0}' "$SUMMARY_TSV")
SKIP_COUNT=$(awk -F'\t' 'NR>1 && ($2=="SKIPPED_DRY_RUN") {c++} END {print c+0}' "$SUMMARY_TSV")

cat "$SUMMARY_TSV"

echo
echo "Result counts: pass=$PASS_COUNT fail=$FAIL_COUNT skipped=$SKIP_COUNT"

if [[ $FAIL_COUNT -gt 0 ]]; then
    exit 1
fi

exit 0