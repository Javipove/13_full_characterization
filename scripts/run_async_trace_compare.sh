#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

RUNNER="$ROOT_DIR/run_full_charac.sh"
BINARY="$ROOT_DIR/build/test_full_charac"
OUTPUT=""
SUMMARY_OUTPUT=""
ASYNC_ITERS=20
TRACE_REPLAY_ITERS=20
NUM_RT_ARGS=8
TRACE_CAPTURE_OPS=1
DRY_RUN=0
USED_LEGACY_ITERS_ALIAS=0

usage() {
    cat <<'USAGE'
Usage:
  ./scripts/run_async_trace_compare.sh [options]

Options:
  --output FILE              Output log file path. Appends results if file exists.
  --summary-output FILE      Parsed summary log path. If omitted, defaults next to --output.
    --async-iters N            Number of async enqueues for Test 5 (default: 20).
    --trace-replay-iters N     Number of trace replays for Test 6 (default: 20).
    --iters N                  Deprecated alias: sets both async and trace replay iters.
  --num-rt-args N            Kernel runtime args for both async/trace runs (default: 8).
  --trace-capture-ops N      Number of ops captured into trace setup phase (default: 1).
  --trace-rt-args N          Deprecated alias for --num-rt-args.
  --runner PATH              Path to run_full_charac.sh (default: ./run_full_charac.sh).
  --binary PATH              Path to test_full_charac binary (default: ./build/test_full_charac).
  --dry-run                  Print commands without executing.
  -h, --help                 Show this help.

What it runs (exact comparison setup):
  - Test 5: ComputeMMAsyncBatch
  - Test 6: ComputeMMTraceReplay
    - Derived op basis:
            async_ops_total = async_iters
            trace_ops_total = trace_capture_ops * trace_replay_iters
            op_matched = (async_ops_total == trace_ops_total)
  - Config matrix:
      1) size=512 mode=dram
      2) size=4096 mode=dram
  - Common flags: --cache --bypass-check --x_size 8 --y_size 7
    - Raw total-time side-by-side comparison is only valid when op_matched=true.

Output behavior:
  - Appends raw command output and parsed timing summary into --output log.
  - Writes a concise per-case parsed summary to --summary-output.
USAGE
}

is_uint() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

to_abs_from_root() {
    local path="$1"
    if [[ "$path" == /* ]]; then
        echo "$path"
    else
        echo "$ROOT_DIR/$path"
    fi
}

extract_num_field() {
    local line="$1"
    local key="$2"
    local suffix="$3"
    echo "$line" | sed -nE "s/.*${key}=([0-9]+)${suffix}.*/\\1/p"
}

extract_text_field() {
    local line="$1"
    local key="$2"
    local next_key="$3"
    echo "$line" | sed -nE "s/.*${key}=([^,]+), ${next_key}=.*/\\1/p"
}

is_uint_value() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --summary-output)
            SUMMARY_OUTPUT="$2"
            shift 2
            ;;
        --async-iters)
            ASYNC_ITERS="$2"
            shift 2
            ;;
        --trace-replay-iters)
            TRACE_REPLAY_ITERS="$2"
            shift 2
            ;;
        --iters)
            ASYNC_ITERS="$2"
            TRACE_REPLAY_ITERS="$2"
            USED_LEGACY_ITERS_ALIAS=1
            shift 2
            ;;
        --num-rt-args)
            NUM_RT_ARGS="$2"
            shift 2
            ;;
        --trace-capture-ops)
            TRACE_CAPTURE_OPS="$2"
            shift 2
            ;;
        --trace-rt-args)
            NUM_RT_ARGS="$2"
            shift 2
            ;;
        --runner)
            RUNNER="$2"
            shift 2
            ;;
        --binary)
            BINARY="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

if ! is_uint "$ASYNC_ITERS" || [[ "$ASYNC_ITERS" -lt 1 ]]; then
    echo "Error: --async-iters must be an integer >= 1"
    exit 1
fi

if ! is_uint "$TRACE_REPLAY_ITERS" || [[ "$TRACE_REPLAY_ITERS" -lt 1 ]]; then
    echo "Error: --trace-replay-iters must be an integer >= 1"
    exit 1
fi

if ! is_uint "$NUM_RT_ARGS" || [[ "$NUM_RT_ARGS" -lt 1 ]]; then
    echo "Error: --num-rt-args must be an integer >= 1"
    exit 1
fi

if ! is_uint "$TRACE_CAPTURE_OPS" || [[ "$TRACE_CAPTURE_OPS" -lt 1 ]]; then
    echo "Error: --trace-capture-ops must be an integer >= 1"
    exit 1
fi

if [[ "$USED_LEGACY_ITERS_ALIAS" -eq 1 ]]; then
    echo "WARN: --iters is deprecated; use --async-iters and --trace-replay-iters for modular control."
fi

RUNNER="$(to_abs_from_root "$RUNNER")"
BINARY="$(to_abs_from_root "$BINARY")"

if [[ -z "$OUTPUT" ]]; then
    OUTPUT="$ROOT_DIR/logs/async_trace_comparison.log"
else
    OUTPUT="$(to_abs_from_root "$OUTPUT")"
fi

if [[ -z "$SUMMARY_OUTPUT" ]]; then
    if [[ "$OUTPUT" == *.log ]]; then
        SUMMARY_OUTPUT="${OUTPUT%.log}.summary.log"
    else
        SUMMARY_OUTPUT="${OUTPUT}.summary.log"
    fi
else
    SUMMARY_OUTPUT="$(to_abs_from_root "$SUMMARY_OUTPUT")"
fi

mkdir -p "$(dirname "$OUTPUT")"
mkdir -p "$(dirname "$SUMMARY_OUTPUT")"

if [[ "$DRY_RUN" -eq 0 ]]; then
    if [[ ! -x "$RUNNER" ]]; then
        echo "Error: runner not found or not executable: $RUNNER"
        exit 1
    fi
    if [[ ! -x "$BINARY" ]]; then
        echo "Error: binary not found or not executable: $BINARY"
        exit 1
    fi
fi

export TT_METAL_HOME="${TT_METAL_HOME:-/scratch/javier/tt-metal}"
export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-$TT_METAL_HOME}"

{
    echo ""
    echo "################################################################################"
    echo "# Async vs Trace comparison run"
    echo "# timestamp=$(date +%Y-%m-%dT%H:%M:%S%z)"
    echo "# runner=$RUNNER"
    echo "# binary=$BINARY"
    echo "# async_iters=$ASYNC_ITERS"
    echo "# trace_replay_iters=$TRACE_REPLAY_ITERS"
    echo "# num_rt_args=$NUM_RT_ARGS"
    echo "# trace_capture_ops=$TRACE_CAPTURE_OPS"
    echo "# async_ops_total=$ASYNC_ITERS"
    echo "# trace_ops_total=$((TRACE_CAPTURE_OPS * TRACE_REPLAY_ITERS))"
    echo "# TT_METAL_HOME=$TT_METAL_HOME"
    echo "################################################################################"
} >> "$OUTPUT"

{
    echo "################################################################################"
    echo "# Async vs Trace parsed summary (hierarchical)"
    echo "# timestamp: $(date +%Y-%m-%dT%H:%M:%S%z)"
    echo "# raw_output: $OUTPUT"
    echo "################################################################################"
    echo ""
    echo "Metric glossary:"
    echo "- async_enqueue_us: Host-side time to enqueue async batch work (lower is better)."
    echo "- async_finish_us: Host-side wait time spent in final synchronization for async test."
    echo "- async_total_us: Async execution-phase timing (enqueue + finish) for X ops."
    echo "- trace_capture_ops: Number of ops recorded in one trace capture setup phase."
    echo "- trace_capture_us: One-time trace capture/record overhead."
    echo "- trace_issue_us: Host-side replay issue overhead for trace (lower is better)."
    echo "- trace_finish_us: Host-side wait during replay completion."
    echo "- trace_total_us: Trace execution-phase timing (replay issue + finish)."
    echo "- trace_total_for_x_us: Trace total cost for replayed ops = trace_capture_us + trace_total_us."
    echo "- async_ops_total: Total async ops in execution phase (= async_iters)."
    echo "- trace_ops_total: Total trace replayed ops (= trace_capture_ops * trace_replay_iters)."
    echo "- op_matched: true only when async_ops_total == trace_ops_total."
    echo "- per_op_* metrics normalize by executed ops and remain valid even when op_matched=false."
    echo "- raw total-time comparisons are only side-by-side fair when op_matched=true."
    echo ""
} > "$SUMMARY_OUTPUT"

run_case() {
    local size="$1"
    local mode="$2"
    local dram_flag=""

    if [[ "$mode" == "dram" ]]; then
        dram_flag="--dram"
    fi

    local header="===== COMPARE async_vs_trace | test5=ComputeMMAsyncBatch | test6=ComputeMMTraceReplay | size=${size} | mode=${mode} | async_iters=${ASYNC_ITERS} | trace_replay_iters=${TRACE_REPLAY_ITERS} | num_rt_args=${NUM_RT_ARGS} | trace_capture_ops=${TRACE_CAPTURE_OPS} | cache=on | bypass_check=on ====="

    echo "$header" | tee -a "$OUTPUT"

    local cmd5="$RUNNER $BINARY --test 5 --num-iters ${ASYNC_ITERS} --num-rt-args ${NUM_RT_ARGS} --x_size 8 --y_size 7 --m ${size} --n ${size} --k ${size} --cache ${dram_flag} --bypass-check --pack-tile device --unpack-tile device --input-dtype bfp8 --output-dtype native --cpu 0 --cpu-range 4"
    local cmd6="$RUNNER $BINARY --test 6 --num-iters ${TRACE_REPLAY_ITERS} --num-rt-args ${NUM_RT_ARGS} --trace-capture-ops ${TRACE_CAPTURE_OPS} --x_size 8 --y_size 7 --m ${size} --n ${size} --k ${size} --cache ${dram_flag} --bypass-check --pack-tile device --unpack-tile device --input-dtype bfp8 --output-dtype native --cpu 0 --cpu-range 4"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "CMD: $cmd5" | tee -a "$OUTPUT"
        echo "CMD: $cmd6" | tee -a "$OUTPUT"
        echo "--- PARSED SUMMARY size=${size} mode=${mode} ---" | tee -a "$OUTPUT"
        echo "(dry-run: no execution)" | tee -a "$OUTPUT"
        echo "" | tee -a "$OUTPUT"
        {
            echo "================================================================================"
            echo "Experiment: size=${size}, mode=${mode}, async_iters=${ASYNC_ITERS}, trace_replay_iters=${TRACE_REPLAY_ITERS}, num_rt_args=${NUM_RT_ARGS}, trace_capture_ops=${TRACE_CAPTURE_OPS}, status=SKIPPED_DRY_RUN"
            echo "  Async (Test 5): skipped"
            echo "  Trace (Test 6): skipped"
            echo ""
        } >> "$SUMMARY_OUTPUT"
        return
    fi

    local tmp5 tmp6
    tmp5="$(mktemp)"
    tmp6="$(mktemp)"
    trap 'rm -f "$tmp5" "$tmp6"' RETURN

    echo "CMD: $cmd5" | tee -a "$OUTPUT"
    set +e
    bash -lc "$cmd5" 2>&1 | tee "$tmp5" | tee -a "$OUTPUT"
    local rc5=${PIPESTATUS[0]}

    echo "CMD: $cmd6" | tee -a "$OUTPUT"
    bash -lc "$cmd6" 2>&1 | tee "$tmp6" | tee -a "$OUTPUT"
    local rc6=${PIPESTATUS[0]}
    set -e

    echo "--- PARSED SUMMARY size=${size} mode=${mode} ---" | tee -a "$OUTPUT"
    grep -E "Async batch timing:" "$tmp5" | tee -a "$OUTPUT" || true
    grep -E "Trace replay config:|Trace timing: capture_record_window=|Trace timing: replay_issue_window=" "$tmp6" | tee -a "$OUTPUT" || true

    local async_line trace_cfg_line trace_capture_line trace_replay_line tile_line
    async_line="$(grep -m1 -E "Async batch timing:" "$tmp5" || true)"
    trace_cfg_line="$(grep -m1 -E "Trace replay config:" "$tmp6" || true)"
    trace_capture_line="$(grep -m1 -E "Trace timing: capture_record_window=" "$tmp6" || true)"
    trace_replay_line="$(grep -m1 -E "Trace timing: replay_issue_window=" "$tmp6" || true)"
    tile_line="$(grep -m1 -E "pack_tile=.*unpack_tile=" "$tmp5" || true)"

    local async_enqueue_us="NA" async_finish_us="NA" async_total_us="NA"
    local trace_capture_ops="NA" trace_capture_us="NA" trace_issue_us="NA" trace_finish_us="NA" trace_total_us="NA"
    local pack_tile="NA" unpack_tile="NA"

    if [[ -n "$async_line" ]]; then
        async_enqueue_us="$(extract_num_field "$async_line" "enqueue_window" "us")"
        async_finish_us="$(extract_num_field "$async_line" "finish_wait" "us")"
        async_total_us="$(extract_num_field "$async_line" "total" "us")"
        async_enqueue_us="${async_enqueue_us:-NA}"
        async_finish_us="${async_finish_us:-NA}"
        async_total_us="${async_total_us:-NA}"
    fi

    if [[ -n "$trace_cfg_line" ]]; then
        trace_capture_ops="$(extract_num_field "$trace_cfg_line" "capture_ops_per_trace" "")"
        trace_capture_ops="${trace_capture_ops:-NA}"
    fi

    if [[ -n "$trace_capture_line" ]]; then
        trace_capture_us="$(extract_num_field "$trace_capture_line" "capture_record_window" "us")"
        trace_capture_us="${trace_capture_us:-NA}"
    fi

    if [[ -n "$trace_replay_line" ]]; then
        trace_issue_us="$(extract_num_field "$trace_replay_line" "replay_issue_window" "us")"
        trace_finish_us="$(extract_num_field "$trace_replay_line" "finish_wait" "us")"
        trace_total_us="$(extract_num_field "$trace_replay_line" "replay_total" "us")"
        trace_issue_us="${trace_issue_us:-NA}"
        trace_finish_us="${trace_finish_us:-NA}"
        trace_total_us="${trace_total_us:-NA}"
    fi

    if [[ -n "$tile_line" ]]; then
        pack_tile="$(extract_text_field "$tile_line" "pack_tile" "unpack_tile")"
        unpack_tile="$(echo "$tile_line" | sed -nE 's/.*unpack_tile=([^, ]+).*/\1/p')"
        pack_tile="${pack_tile:-NA}"
        unpack_tile="${unpack_tile:-NA}"
    fi

    if [[ "$trace_capture_ops" == "NA" ]]; then
        trace_capture_ops="$TRACE_CAPTURE_OPS"
    fi

    local status="PASS"
    if [[ "$rc5" -ne 0 || "$rc6" -ne 0 ]]; then
        status="FAIL_CMD"
    elif grep -Eiq "error|critical" "$tmp5" "$tmp6"; then
        status="WARN_LOG_MARKERS"
    fi

    local async_ops_total="$ASYNC_ITERS"
    local trace_ops_total="NA"
    if is_uint_value "$trace_capture_ops"; then
        trace_ops_total=$((TRACE_REPLAY_ITERS * trace_capture_ops))
    fi

    local op_matched="false"
    if is_uint_value "$trace_ops_total" && [[ "$async_ops_total" -eq "$trace_ops_total" ]]; then
        op_matched="true"
    fi

    local fair_side_by_side="$op_matched"

    local async_ops="$ASYNC_ITERS"
    local trace_replayed_ops="NA"
    if is_uint_value "$trace_capture_ops"; then
        trace_replayed_ops=$((TRACE_REPLAY_ITERS * trace_capture_ops))
    fi

    local async_per_op_exec_us="NA"
    local trace_per_op_exec_us="NA"
    local trace_total_for_x_us="NA"
    local trace_per_op_total_with_capture_us="NA"
    local delta_exec_us="NA"
    local delta_total_with_capture_us="NA"
    local ratio_trace_exec_over_async_exec="NA"
    local ratio_trace_total_over_async_exec="NA"

    if is_uint_value "$async_total_us" && [[ "$async_ops" -gt 0 ]]; then
        async_per_op_exec_us="$(awk -v t="$async_total_us" -v n="$async_ops" 'BEGIN { printf "%.4f", t / n }')"
    fi

    if is_uint_value "$trace_total_us" && is_uint_value "$trace_replayed_ops" && [[ "$trace_replayed_ops" -gt 0 ]]; then
        trace_per_op_exec_us="$(awk -v t="$trace_total_us" -v n="$trace_replayed_ops" 'BEGIN { printf "%.4f", t / n }')"
    fi

    if is_uint_value "$trace_capture_us" && is_uint_value "$trace_total_us"; then
        trace_total_for_x_us=$((trace_capture_us + trace_total_us))
        if is_uint_value "$trace_replayed_ops" && [[ "$trace_replayed_ops" -gt 0 ]]; then
            trace_per_op_total_with_capture_us="$(awk -v t="$trace_total_for_x_us" -v n="$trace_replayed_ops" 'BEGIN { printf "%.4f", t / n }')"
        fi
    fi

    if is_uint_value "$async_total_us" && is_uint_value "$trace_total_us"; then
        delta_exec_us=$((trace_total_us - async_total_us))
        if [[ "$async_total_us" -gt 0 ]]; then
            ratio_trace_exec_over_async_exec="$(awk -v t="$trace_total_us" -v a="$async_total_us" 'BEGIN { printf "%.4f", t / a }')"
        fi
    fi

    if is_uint_value "$async_total_us" && is_uint_value "$trace_total_for_x_us"; then
        delta_total_with_capture_us=$((trace_total_for_x_us - async_total_us))
        if [[ "$async_total_us" -gt 0 ]]; then
            ratio_trace_total_over_async_exec="$(awk -v t="$trace_total_for_x_us" -v a="$async_total_us" 'BEGIN { printf "%.4f", t / a }')"
        fi
    fi

    {
        echo "================================================================================"
        echo "Experiment: size=${size}, mode=${mode}, async_iters=${ASYNC_ITERS}, trace_replay_iters=${TRACE_REPLAY_ITERS}, num_rt_args=${NUM_RT_ARGS}, trace_capture_ops=${trace_capture_ops}, status=${status}"
        echo ""
        echo "Async (Test 5) [Execution Phase]"
        echo "  rc: ${rc5}"
        echo "  ops_executed: ${async_ops}"
        echo "  async_enqueue_us: ${async_enqueue_us}"
        echo "  async_finish_us: ${async_finish_us}"
        echo "  async_total_us: ${async_total_us}"
        echo "  async_per_op_exec_us: ${async_per_op_exec_us}"
        echo ""
        echo "Trace (Test 6) [Execution Phase]"
        echo "  rc: ${rc6}"
        echo "  trace_capture_ops: ${trace_capture_ops}"
        echo "  trace_replayed_ops: ${trace_replayed_ops}"
        echo "  trace_compare_ops_basis: ${TRACE_REPLAY_ITERS}"
        echo "  trace_capture_us: ${trace_capture_us}"
        echo "  trace_issue_us: ${trace_issue_us}"
        echo "  trace_finish_us: ${trace_finish_us}"
        echo "  trace_total_us: ${trace_total_us}"
        echo "  trace_per_op_exec_us: ${trace_per_op_exec_us}"
        echo ""
        echo "Trace (Test 6) [Total Cost For Replayed Ops]"
        echo "  trace_total_for_x_us: ${trace_total_for_x_us}"
        echo "  trace_per_op_total_with_capture_us: ${trace_per_op_total_with_capture_us}"
        echo ""
        echo "Shared"
        echo "  pack_tile: ${pack_tile}"
        echo "  unpack_tile: ${unpack_tile}"
        echo "  async_ops_total: ${async_ops_total}"
        echo "  trace_ops_total: ${trace_ops_total}"
        echo "  op_matched: ${op_matched}"
        echo "  fair_side_by_side: ${fair_side_by_side}"
        echo ""
        echo "Comparison"
        echo "  delta_exec_us(trace_exec-async_exec): ${delta_exec_us}"
        echo "  delta_total_with_capture_us(trace_total_for_x-async_exec): ${delta_total_with_capture_us}"
        echo "  ratio_trace_exec_over_async_exec: ${ratio_trace_exec_over_async_exec}"
        echo "  ratio_trace_total_over_async_exec: ${ratio_trace_total_over_async_exec}"
        if [[ "$op_matched" != "true" ]]; then
            echo "  warning_raw_totals: op_mismatch_raw_totals_not_side_by_side_fair"
        fi
        echo ""
    } >> "$SUMMARY_OUTPUT"

    if grep -Eiq "error|critical" "$tmp5" "$tmp6"; then
        echo "WARN: detected error/critical marker in logs for size=${size} mode=${mode}" | tee -a "$OUTPUT"
    fi

    echo "" | tee -a "$OUTPUT"

    rm -f "$tmp5" "$tmp6"
    trap - RETURN
}

# run_case 512 l1
run_case 512 dram
run_case 4096 dram

echo "Saved comparison log to $OUTPUT"
echo "Saved parsed summary to $SUMMARY_OUTPUT"
