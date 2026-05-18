#!/usr/bin/env bash

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

to_abs_from_root() {
    local path="$1"
    if [[ "$path" == /* ]]; then
        echo "$path"
    else
        echo "$ROOT_DIR/$path"
    fi
}

is_uint() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

safe_tag() {
    echo "$1" | tr ', ' '__' | tr -cd 'A-Za-z0-9._-'
}

split_csv() {
    local csv="$1"
    local out_name="$2"
    local -n out_ref="$out_name"
    IFS=',' read -r -a out_ref <<< "$csv"
    for i in "${!out_ref[@]}"; do
        out_ref[$i]="$(echo "${out_ref[$i]}" | xargs)"
    done
}

validate_uint_list() {
    local name="$1"
    shift
    local values=("$@")
    if [[ ${#values[@]} -eq 0 ]]; then
        echo "Error: --${name} list is empty"
        exit 1
    fi
    for v in "${values[@]}"; do
        if [[ -z "$v" ]] || ! [[ "$v" =~ ^[0-9]+$ ]]; then
            echo "Error: --${name} contains non-integer value '$v'"
            exit 1
        fi
    done
}

validate_binary_mode_list() {
    local name="$1"
    shift
    local values=("$@")
    if [[ ${#values[@]} -eq 0 ]]; then
        echo "Error: --${name} list is empty"
        exit 1
    fi
    for v in "${values[@]}"; do
        if [[ "$v" != "0" && "$v" != "1" ]]; then
            echo "Error: --${name} only accepts 0 or 1 (got '$v')"
            exit 1
        fi
    done
}

tsv_to_csv() {
    local tsv_file="$1"
    local csv_file="$2"
    awk -F'\t' '
        BEGIN { OFS = "," }
        {
            for (i = 1; i <= NF; i++) {
                gsub(/"/, "\"\"", $i)
                $i = "\"" $i "\""
            }
            print $0
        }
    ' "$tsv_file" > "$csv_file"
}

BINARY="build/test_full_charac"
RUNNER="run_full_charac.sh"
TT_METAL_HOME_DEFAULT="$ROOT_DIR"

TEST_IDS="1"
DRAM_MODES="1"
X_SIZES="7"
Y_SIZES="7"
M_SIZES="4096"
N_SIZES="4096"
K_SIZES="4096"
NUM_ITERS_LIST="20"
NUM_RT_ARGS_LIST="255"
CLEAN_MODES="0"
CPU=2
CPU_RANGE=8
INTERNAL_CPU_PIN=0
REPEATS=1
BYPASS_CHECK=1
RESET_DEVICE=1
EXTRA_ARGS=""
TASKSET_CPUS=""
NUMACTL_CPUBIND=""
NUMACTL_MEMBIND=""

CAPTURE_BIN=""
CAPTURE_CMD_TEMPLATE='"{capture_bin}" -o "{output}"'
START_DELAY_SEC=1
STOP_DELAY_SEC=1

OUT_DIR=""
LOG_DIR=""
TRACE_DIR=""
OUTPUT_LOG=""
NO_TIMESTAMP=0
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage:
  ./run_full_charac_tracy_sweep.sh [options]

Execution options:
  --binary PATH                    test_full_charac binary path (default: build/test_full_charac)
  --runner PATH                    run_full_charac.sh path (default: run_full_charac.sh)
  --test-ids LIST                  Comma list of --test ids (default: 1)
  --dram-modes LIST                Comma list of 0/1 (default: 1)
  --x-sizes LIST                   Comma list (default: 7)
  --y-sizes LIST                   Comma list (default: 7)
  --m-sizes LIST                   Comma list (default: 4096)
  --n-sizes LIST                   Comma list (default: 4096)
  --k-sizes LIST                   Comma list (default: 4096)
  --num-iters LIST                 Comma list (default: 20)
  --num-rt-args LIST               Comma list (default: 255)
    --clean-modes LIST               Comma list of 0/1 (default: 0)
  --cpu N                          test_full_charac --cpu value (default: 2)
  --cpu-range N                    test_full_charac --cpu-range value (default: 8)
    --internal-cpu-pin               Pass --cpu/--cpu-range to test_full_charac (disabled by default)
  --repeats N                      Repeat each config N times (default: 1)
  --no-bypass-check                Do not pass --bypass-check
  --no-reset                       Disable tt-smi reset before each command execution
    --taskset-cpus LIST              Pin host process to CPU list/range (for example 2 or 2-8)
    --numactl-cpubind NODE           Bind execution to NUMA node CPUs
    --numactl-membind NODE           Bind memory allocation to NUMA node
  --extra-args STRING              Extra args appended to each command

Tracy capture options:
  --capture-bin PATH               Capture binary path (default: $TT_METAL_HOME/build/tools/profiler/bin/capture-release)
  --capture-cmd-template STRING    Template with placeholders: {capture_bin} {output} {case}
                                   default: '"{capture_bin}" -o "{output}"'
  --start-delay-sec N              Sleep after starting capture (default: 1)
  --stop-delay-sec N               Sleep after capture Ctrl-C (default: 1)

Output options:
    --output FILE                   Aggregate all run output into one .log file
  --out-dir DIR                    Output directory path
  --log-dir DIR                    Log directory path
  --trace-dir DIR                  Trace output directory path
  --no-timestamp                   Deterministic output dir when --out-dir is not provided
  --dry-run                        Print commands only
  -h, --help                       Show this help

Notes:
  - One Tracy file is generated per command execution (including repeated runs).
  - Trace filenames include full case config and repetition index.
    - Internal test pinning (--cpu/--cpu-range) is disabled by default.
    - Prefer --taskset-cpus / --numactl-* for manual host pinning.

Examples:
    # 1) Repeat same config 3 times (3 Tracy files)
    ./run_full_charac_tracy_sweep.sh \
        --test-ids 1 --dram-modes 1 --clean-modes 0 \
        --num-iters 10 --num-rt-args 255 --repeats 3

    # 2) Sweep runtime args + clean mode
    ./run_full_charac_tracy_sweep.sh \
        --test-ids 1 --dram-modes 1 --clean-modes 0,1 \
        --num-rt-args 8,32,128,255 --num-iters 10 --repeats 1

    # 3) Dry run to verify generated command matrix
    ./run_full_charac_tracy_sweep.sh \
        --dry-run --num-rt-args 8,255 --clean-modes 0,1

    # 4) Opt in to internal test pinning (if needed)
    ./run_full_charac_tracy_sweep.sh \
        --internal-cpu-pin --cpu 2 --cpu-range 8
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --binary)
            BINARY="$2"
            shift 2
            ;;
        --runner)
            RUNNER="$2"
            shift 2
            ;;
        --test-ids)
            TEST_IDS="$2"
            shift 2
            ;;
        --dram-modes)
            DRAM_MODES="$2"
            shift 2
            ;;
        --x-sizes)
            X_SIZES="$2"
            shift 2
            ;;
        --y-sizes)
            Y_SIZES="$2"
            shift 2
            ;;
        --m-sizes)
            M_SIZES="$2"
            shift 2
            ;;
        --n-sizes)
            N_SIZES="$2"
            shift 2
            ;;
        --k-sizes)
            K_SIZES="$2"
            shift 2
            ;;
        --num-iters)
            NUM_ITERS_LIST="$2"
            shift 2
            ;;
        --num-rt-args)
            NUM_RT_ARGS_LIST="$2"
            shift 2
            ;;
        --clean-modes)
            CLEAN_MODES="$2"
            shift 2
            ;;
        --cpu)
            CPU="$2"
            shift 2
            ;;
        --cpu-range)
            CPU_RANGE="$2"
            shift 2
            ;;
        --internal-cpu-pin)
            INTERNAL_CPU_PIN=1
            shift
            ;;
        --repeats)
            REPEATS="$2"
            shift 2
            ;;
        --no-bypass-check)
            BYPASS_CHECK=0
            shift
            ;;
        --no-reset)
            RESET_DEVICE=0
            shift
            ;;
        --taskset-cpus)
            TASKSET_CPUS="$2"
            shift 2
            ;;
        --numactl-cpubind)
            NUMACTL_CPUBIND="$2"
            shift 2
            ;;
        --numactl-membind)
            NUMACTL_MEMBIND="$2"
            shift 2
            ;;
        --extra-args)
            EXTRA_ARGS="$2"
            shift 2
            ;;
        --capture-bin)
            CAPTURE_BIN="$2"
            shift 2
            ;;
        --capture-cmd-template)
            CAPTURE_CMD_TEMPLATE="$2"
            shift 2
            ;;
        --start-delay-sec)
            START_DELAY_SEC="$2"
            shift 2
            ;;
        --stop-delay-sec)
            STOP_DELAY_SEC="$2"
            shift 2
            ;;
        --out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --trace-dir)
            TRACE_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT_LOG="$2"
            shift 2
            ;;
        --no-timestamp)
            NO_TIMESTAMP=1
            shift
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

if ! is_uint "$CPU" || ! is_uint "$CPU_RANGE" || ! is_uint "$REPEATS" || ! is_uint "$START_DELAY_SEC" || ! is_uint "$STOP_DELAY_SEC"; then
    echo "Error: --cpu, --cpu-range, --repeats, --start-delay-sec, and --stop-delay-sec must be integers"
    exit 1
fi
if [[ "$REPEATS" -lt 1 ]]; then
    echo "Error: --repeats must be >= 1"
    exit 1
fi
if [[ "$INTERNAL_CPU_PIN" != "0" && "$INTERNAL_CPU_PIN" != "1" ]]; then
    echo "Error: internal CPU pin flag is invalid"
    exit 1
fi

export TT_METAL_HOME="${TT_METAL_HOME:-$TT_METAL_HOME_DEFAULT}"
export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-$TT_METAL_HOME}"
unset TT_METAL_SLOW_DISPATCH_MODE

if [[ "$DRY_RUN" -eq 0 && ! -d "$TT_METAL_HOME/tt_metal/soc_descriptors" ]]; then
    echo "Error: TT_METAL_HOME does not contain runtime descriptors: $TT_METAL_HOME/tt_metal/soc_descriptors"
    echo "Set TT_METAL_HOME to your tt-metal checkout before running this script."
    exit 1
fi

if [[ -z "$CAPTURE_BIN" ]]; then
    CAPTURE_BIN="$TT_METAL_HOME/build/tools/profiler/bin/capture-release"
fi

BINARY="$(to_abs_from_root "$BINARY")"
RUNNER="$(to_abs_from_root "$RUNNER")"
if [[ -n "$OUT_DIR" ]]; then
    OUT_DIR="$(to_abs_from_root "$OUT_DIR")"
fi
if [[ -n "$LOG_DIR" ]]; then
    LOG_DIR="$(to_abs_from_root "$LOG_DIR")"
fi
if [[ -n "$TRACE_DIR" ]]; then
    TRACE_DIR="$(to_abs_from_root "$TRACE_DIR")"
fi
if [[ -n "$OUTPUT_LOG" ]]; then
    OUTPUT_LOG="$(to_abs_from_root "$OUTPUT_LOG")"
fi
if [[ "$CAPTURE_BIN" != /* ]]; then
    CAPTURE_BIN="$(to_abs_from_root "$CAPTURE_BIN")"
fi

if [[ ! -x "$RUNNER" ]]; then
    echo "Error: runner not found or not executable: $RUNNER"
    exit 1
fi
if [[ ! -x "$BINARY" ]]; then
    echo "Error: binary not found or not executable: $BINARY"
    exit 1
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
    if [[ -n "$TASKSET_CPUS" ]] && ! command -v taskset >/dev/null 2>&1; then
        echo "Error: --taskset-cpus requested but taskset is not available in PATH"
        exit 1
    fi
    if [[ -n "$NUMACTL_CPUBIND" || -n "$NUMACTL_MEMBIND" ]]; then
        if ! command -v numactl >/dev/null 2>&1; then
            echo "Error: NUMA binding requested but numactl is not available in PATH"
            exit 1
        fi
    fi
fi

EXEC_PREFIX=""
if [[ -n "$NUMACTL_CPUBIND" || -n "$NUMACTL_MEMBIND" ]]; then
    EXEC_PREFIX+="numactl"
    if [[ -n "$NUMACTL_CPUBIND" ]]; then
        EXEC_PREFIX+=" --cpubind=${NUMACTL_CPUBIND}"
    fi
    if [[ -n "$NUMACTL_MEMBIND" ]]; then
        EXEC_PREFIX+=" --membind=${NUMACTL_MEMBIND}"
    fi
    EXEC_PREFIX+=" "
fi
if [[ -n "$TASKSET_CPUS" ]]; then
    EXEC_PREFIX+="taskset -c ${TASKSET_CPUS} "
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
    if ! command -v tmux >/dev/null 2>&1; then
        echo "Error: tmux is not available in PATH"
        exit 1
    fi
    if [[ "$RESET_DEVICE" -eq 1 ]] && ! command -v tt-smi >/dev/null 2>&1; then
        echo "Error: --no-reset not set and tt-smi is not available in PATH"
        exit 1
    fi
    if [[ ! -x "$CAPTURE_BIN" ]]; then
        echo "Error: capture binary not found or not executable: $CAPTURE_BIN"
        echo "Hint: set --capture-bin or export TT_METAL_HOME to your tt-metal root."
        exit 1
    fi
fi

if [[ -n "$TASKSET_CPUS" && "$INTERNAL_CPU_PIN" -eq 1 ]]; then
    echo "Warning: both taskset pinning and internal --cpu/--cpu-range pinning are enabled."
    echo "         This may not map 1:1; if needed use --no-internal-cpu-pin."
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -z "$OUT_DIR" ]]; then
    BASE="full_charac_tracy_t$(safe_tag "$TEST_IDS")_d$(safe_tag "$DRAM_MODES")_c$(safe_tag "$CLEAN_MODES")_x$(safe_tag "$X_SIZES")_y$(safe_tag "$Y_SIZES")_m$(safe_tag "$M_SIZES")_n$(safe_tag "$N_SIZES")_k$(safe_tag "$K_SIZES")_it$(safe_tag "$NUM_ITERS_LIST")_rt$(safe_tag "$NUM_RT_ARGS_LIST")_r${REPEATS}"
    if [[ "$NO_TIMESTAMP" -eq 1 ]]; then
        OUT_DIR="$ROOT_DIR/logs/${BASE}"
    else
        OUT_DIR="$ROOT_DIR/logs/${BASE}_${STAMP}"
    fi
fi
if [[ -z "$LOG_DIR" ]]; then
    LOG_DIR="$OUT_DIR/logs"
fi
if [[ -z "$TRACE_DIR" ]]; then
    TRACE_DIR="$OUT_DIR/traces"
fi
if [[ -z "$OUTPUT_LOG" ]]; then
    OUTPUT_LOG="$OUT_DIR/sweep_output.log"
fi

mkdir -p "$OUT_DIR" "$LOG_DIR" "$TRACE_DIR"
mkdir -p "$(dirname "$OUTPUT_LOG")"

{
    echo "==== full_charac_tracy_sweep ===="
    echo "timestamp=${STAMP}"
    echo "out_dir=${OUT_DIR}"
    echo "log_dir=${LOG_DIR}"
    echo "trace_dir=${TRACE_DIR}"
    echo "output_log=${OUTPUT_LOG}"
    echo "runner=${RUNNER}"
    echo "binary=${BINARY}"
    echo "repeats=${REPEATS}"
    echo "dry_run=${DRY_RUN}"
    echo ""
} > "$OUTPUT_LOG"

MEASUREMENTS_TSV="$OUT_DIR/measurements.tsv"
SUMMARY_TSV="$OUT_DIR/summary.tsv"
MEASUREMENTS_CSV="$OUT_DIR/measurements.csv"
SUMMARY_CSV="$OUT_DIR/summary.csv"

printf "idx\trep\tcase_tag\ttest_id\tdram\tclean_mode\tx\ty\tm\tn\tk\tnum_iters\tnum_rt_args\tstatus\texit_code\tcapture_status\ttrace_file\tlog_file\tcommand\tcapture_command\n" > "$MEASUREMENTS_TSV"
printf "idx\tcase_tag\trepeats\tstatus\tfail_reps\tcapture_fail_reps\ttrace_dir\tlast_log_file\tcommand\n" > "$SUMMARY_TSV"

split_csv "$TEST_IDS" TEST_LIST
split_csv "$DRAM_MODES" DRAM_LIST
split_csv "$X_SIZES" X_LIST
split_csv "$Y_SIZES" Y_LIST
split_csv "$M_SIZES" M_LIST
split_csv "$N_SIZES" N_LIST
split_csv "$K_SIZES" K_LIST
split_csv "$NUM_ITERS_LIST" ITER_LIST
split_csv "$NUM_RT_ARGS_LIST" RTARG_LIST
split_csv "$CLEAN_MODES" CLEAN_MODE_LIST

validate_uint_list "test-ids" "${TEST_LIST[@]}"
validate_binary_mode_list "dram-modes" "${DRAM_LIST[@]}"
validate_binary_mode_list "clean-modes" "${CLEAN_MODE_LIST[@]}"
validate_uint_list "x-sizes" "${X_LIST[@]}"
validate_uint_list "y-sizes" "${Y_LIST[@]}"
validate_uint_list "m-sizes" "${M_LIST[@]}"
validate_uint_list "n-sizes" "${N_LIST[@]}"
validate_uint_list "k-sizes" "${K_LIST[@]}"
validate_uint_list "num-iters" "${ITER_LIST[@]}"
validate_uint_list "num-rt-args" "${RTARG_LIST[@]}"

idx=0
pass_count=0
fail_count=0
skip_count=0

for test_id in "${TEST_LIST[@]}"; do
for dram in "${DRAM_LIST[@]}"; do
for x in "${X_LIST[@]}"; do
for y in "${Y_LIST[@]}"; do
for m in "${M_LIST[@]}"; do
for n in "${N_LIST[@]}"; do
for k in "${K_LIST[@]}"; do
for nit in "${ITER_LIST[@]}"; do
for nrt in "${RTARG_LIST[@]}"; do
for clean_mode in "${CLEAN_MODE_LIST[@]}"; do

    case_tag="t${test_id}_d${dram}_c${clean_mode}_x${x}_y${y}_m${m}_n${n}_k${k}_it${nit}_rt${nrt}"
    base_cmd="$RUNNER $BINARY --test ${test_id} --num-iters ${nit} --num-rt-args ${nrt} --clean-mode ${clean_mode} --x_size ${x} --y_size ${y} --m ${m} --n ${n} --k ${k}"
    if [[ "$INTERNAL_CPU_PIN" -eq 1 ]]; then
        base_cmd+=" --cpu ${CPU} --cpu-range ${CPU_RANGE}"
    fi
    if [[ "$dram" == "1" ]]; then
        base_cmd+=" --dram"
    fi
    if [[ "$BYPASS_CHECK" -eq 1 ]]; then
        base_cmd+=" --bypass-check"
    fi
    if [[ -n "$EXTRA_ARGS" ]]; then
        base_cmd+=" ${EXTRA_ARGS}"
    fi
    cmd="${EXEC_PREFIX}${base_cmd}"

    echo
    echo "[$((idx + 1))] $case_tag"
    echo "Command: $cmd"

    {
        echo "===== CASE idx=$idx tag=$case_tag ====="
        echo "Command: $cmd"
    } >> "$OUTPUT_LOG"

    case_status="PASS"
    case_fail_reps=0
    case_capture_fail_reps=0
    last_log_file=""

    for rep in $(seq 1 "$REPEATS"); do
        trace_file="$TRACE_DIR/$(printf "%04d" "$idx")_${case_tag}_r$(printf "%02d" "$rep").tracy"
        log_file="$LOG_DIR/$(printf "%04d" "$idx")_${case_tag}_r$(printf "%02d" "$rep").log"
        reset_log="$LOG_DIR/$(printf "%04d" "$idx")_${case_tag}_r$(printf "%02d" "$rep")_reset.log"
        capture_log="$LOG_DIR/$(printf "%04d" "$idx")_${case_tag}_r$(printf "%02d" "$rep")_capture.log"

        capture_cmd="$CAPTURE_CMD_TEMPLATE"
        capture_cmd="${capture_cmd//\{capture_bin\}/$CAPTURE_BIN}"
        capture_cmd="${capture_cmd//\{output\}/$trace_file}"
        capture_cmd="${capture_cmd//\{case\}/$case_tag}"

        rc=0
        rep_status="PASS"
        capture_status="ok"

        if [[ "$DRY_RUN" -eq 1 ]]; then
            rep_status="SKIPPED_DRY_RUN"
            capture_status="skipped_dry_run"
            skip_count=$((skip_count + 1))
            {
                echo "--- REP $rep/$REPEATS (dry-run) ---"
                echo "$cmd"
                echo "status=$rep_status"
                echo ""
            } >> "$OUTPUT_LOG"
        else
            session_name="tracy_${idx}_r${rep}_$$"
            tmux new-session -d -s "$session_name" "bash -lc '$capture_cmd'" > "$capture_log" 2>&1 || true
            sleep "$START_DELAY_SEC"

            if [[ "$RESET_DEVICE" -eq 1 ]]; then
                tt-smi -r 0 > "$reset_log" 2>&1 || true
            fi

            {
                echo "--- REP $rep/$REPEATS ---"
                echo "trace_file=$trace_file"
                echo "capture_cmd=$capture_cmd"
                echo "log_file=$log_file"
            } >> "$OUTPUT_LOG"

            bash -lc "$cmd" |& tee "$log_file" | tee -a "$OUTPUT_LOG"
            rc=${PIPESTATUS[0]}

            if tmux has-session -t "$session_name" 2>/dev/null; then
                tmux send-keys -t "$session_name" C-c >/dev/null 2>&1 || true
                sleep "$STOP_DELAY_SEC"
                tmux kill-session -t "$session_name" >/dev/null 2>&1 || true
            fi

            if [[ ! -s "$trace_file" ]]; then
                capture_status="missing_or_empty"
                case_capture_fail_reps=$((case_capture_fail_reps + 1))
            fi

            if [[ "$rc" -ne 0 ]]; then
                rep_status="FAIL"
                case_status="FAIL"
                case_fail_reps=$((case_fail_reps + 1))
            fi

            {
                echo "rep_status=$rep_status"
                echo "exit_code=$rc"
                echo "capture_status=$capture_status"
                echo ""
            } >> "$OUTPUT_LOG"
        fi

        last_log_file="$log_file"

        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$idx" "$rep" "$case_tag" "$test_id" "$dram" "$clean_mode" "$x" "$y" "$m" "$n" "$k" "$nit" "$nrt" \
            "$rep_status" "$rc" "$capture_status" "$trace_file" "$log_file" "$cmd" "$capture_cmd" >> "$MEASUREMENTS_TSV"
    done

    if [[ "$DRY_RUN" -eq 0 ]]; then
        if [[ "$case_status" == "PASS" ]]; then
            pass_count=$((pass_count + 1))
        else
            fail_count=$((fail_count + 1))
        fi
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$idx" "$case_tag" "$REPEATS" "$case_status" "$case_fail_reps" "$case_capture_fail_reps" "$TRACE_DIR" "$last_log_file" "$cmd" >> "$SUMMARY_TSV"

    idx=$((idx + 1))

done
done
done
done
done
done
done
done
done
done

tsv_to_csv "$SUMMARY_TSV" "$SUMMARY_CSV"
tsv_to_csv "$MEASUREMENTS_TSV" "$MEASUREMENTS_CSV"

echo
echo "Summary TSV: $SUMMARY_TSV"
echo "Measurements TSV: $MEASUREMENTS_TSV"
echo "Summary CSV: $SUMMARY_CSV"
echo "Measurements CSV: $MEASUREMENTS_CSV"
echo "Trace directory: $TRACE_DIR"
echo "Log directory: $LOG_DIR"
echo "Output log: $OUTPUT_LOG"
echo "Counts: pass=$pass_count fail=$fail_count skipped=$skip_count"

if [[ "$DRY_RUN" -eq 1 ]]; then
    exit 0
fi
if [[ "$fail_count" -gt 0 ]]; then
    exit 1
fi
exit 0
