#!/usr/bin/env bash

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

BINARY="./build/test_pgm_dispatch"
WARMUP=2000
ITERATIONS=3000
PHASES="A,B,C,D"
MODE="phases"
REPEATS=1
DRY_RUN=0
RESET_BEFORE_EACH=0
RUN_GBENCH=0
GBENCH_FILTER="BM_pgm_dispatch/(all_processors_trace|all_processors_all_cores_trace).*"
CACHE_MODE="warm"
CACHE_DIR=""
TASKSET_CPUS=""
NUMACTL_CPUBIND=""
NUMACTL_MEMBIND=""
OUT_DIR=""
LOG_DIR=""
NO_TIMESTAMP=0

usage() {
    cat <<'EOF'
Usage:
  ./run_pgm_dispatch_host_suite.sh [options]

Options:
  --binary PATH              Path to test_pgm_dispatch binary (default: ./build/test_pgm_dispatch)
    --mode MODE                phases or matrix32 (default: phases)
  --warmup N                 Warmup iterations for --custom runs (default: 2000)
  --iters N                  Timed iterations for --custom runs (default: 3000)
    --repeats N                Repeat each case N times and report mean/std (default: 1)
  --phases LIST              Comma-separated phases to run: A,B,C,D (default: A,B,C,D)
    --cache-mode MODE          warm|clean-start|clean-case (default: warm)
    --cache-dir DIR            Cache directory to clean when cache-mode is not warm
    --out-dir DIR              Output directory path
    --log-dir DIR              Log directory path (cleaned each non-dry run)
    --no-timestamp             Use deterministic output dir name if --out-dir is not set
    --taskset-cpus LIST        Pin process to CPU list/range (for example 2 or 2-8)
    --numactl-cpubind NODE     Bind execution to NUMA node CPUs
    --numactl-membind NODE     Bind memory allocation to NUMA node
    --reset                    Reset device with tt-smi -r 0 before each case/repeat
    --reset-before-each        Alias for --reset
  --run-gbench               Also run non-custom Google Benchmark mode after the custom suite
  --gbench-filter REGEX      Filter used with --run-gbench (default targets trace configs)
  --dry-run                  Print commands only, do not execute
  -h, --help                 Show this help

Phase map:
  A = Minimal baseline (1 core, brisc only, trace/finish pairs)
  B = Runtime-arg sweep (8,32,128,255) with trace/finish pairs
  C = Realistic workload-ish configs (7x7 core range, metadata stress)
  D = Fragmentation sensitivity (kernel groups + subdevice range sweep)
  matrix32 mode = 7x7 vs 1x1 x args(8,64,128,255) x trace(0/1) x finish(0/1) = 32 cases

Notes:
  - This suite targets host dispatch behavior and CMA impact by using --custom mode.
  - It does not use built-in golden JSON comparisons.
  - Parsed numeric metrics are written to measurements.tsv and summary.tsv.
    - CSV exports are auto-generated: summary.csv and measurements.csv.
    - In matrix32 mode, a pivot-friendly matrix32_pivot.csv is also generated.
    - Log files are overwritten each run in the selected log directory.
    - If --out-dir is not set, output defaults to logs/pgm_dispatch_suite_<config>_<timestamp>.
    - Cache cleaning can be controlled with --cache-mode for warm vs cold studies.
EOF
}

is_uint() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

safe_tag() {
    echo "$1" | tr ', ' '__' | tr -cd 'A-Za-z0-9._-'
}

extract_total_us() {
    local log_file="$1"
    awk '
        match($0,/Ran in ([0-9]+(\.[0-9]+)?)us/,m) && $0 !~ /per iteration/ {v=m[1]}
        END{if(v=="") print "NA"; else print v}
    ' "$log_file"
}

extract_per_iter_us() {
    local log_file="$1"
    awk '
        match($0,/Ran in ([0-9]+(\.[0-9]+)?)us per iteration/,m){v=m[1]}
        END{if(v=="") print "NA"; else print v}
    ' "$log_file"
}

compute_mean_std() {
    local metrics_file="$1"
    local field_index="$2"
    awk -F'\t' -v idx="$field_index" '
        $idx != "NA" && $idx != "" {v[n++]=$idx; s+=$idx; ss+=$idx*$idx}
        END {
            if (n == 0) {
                print "NA\tNA"
                exit
            }
            mean = s / n
            var = (ss / n) - (mean * mean)
            if (var < 0) var = 0
            std = sqrt(var)
            printf "%.6f\t%.6f\n", mean, std
        }
    ' "$metrics_file"
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

generate_matrix32_pivot_csv() {
    local summary_tsv="$1"
    local pivot_csv="$2"
    awk -F'\t' '
        BEGIN {
            OFS = ","
            print "grid,runtime_args,tr0_f0,tr0_f1,tr1_f0,tr1_f1"
        }
        NR == 1 { next }
        {
            grid = $4
            args = $5
            trace = $6
            finish = $7
            val = $14
            key = grid "|" args
            vals[key, trace, finish] = val
            seen[key] = 1
        }
        END {
            for (k in seen) {
                split(k, p, "|")
                print p[1], p[2], vals[k,0,0], vals[k,0,1], vals[k,1,0], vals[k,1,1]
            }
        }
    ' "$summary_tsv" > "$pivot_csv"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --binary)
            BINARY="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --warmup)
            WARMUP="$2"
            shift 2
            ;;
        --iters)
            ITERATIONS="$2"
            shift 2
            ;;
        --repeats)
            REPEATS="$2"
            shift 2
            ;;
        --phases)
            PHASES="$2"
            shift 2
            ;;
        --cache-mode)
            CACHE_MODE="$2"
            shift 2
            ;;
        --cache-dir)
            CACHE_DIR="$2"
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
        --no-timestamp)
            NO_TIMESTAMP=1
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
        --reset-before-each)
            RESET_BEFORE_EACH=1
            shift
            ;;
        --reset)
            RESET_BEFORE_EACH=1
            shift
            ;;
        --run-gbench)
            RUN_GBENCH=1
            shift
            ;;
        --gbench-filter)
            GBENCH_FILTER="$2"
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

if [[ "$MODE" != "phases" && "$MODE" != "matrix32" ]]; then
    echo "Error: --mode must be one of: phases, matrix32"
    exit 1
fi

if [[ "$CACHE_MODE" != "warm" && "$CACHE_MODE" != "clean-start" && "$CACHE_MODE" != "clean-case" ]]; then
    echo "Error: --cache-mode must be one of: warm, clean-start, clean-case"
    exit 1
fi

if ! is_uint "$WARMUP" || ! is_uint "$ITERATIONS" || ! is_uint "$REPEATS"; then
    echo "Error: --warmup, --iters, and --repeats must be non-negative integers"
    exit 1
fi

if [[ "$REPEATS" -lt 1 ]]; then
    echo "Error: --repeats must be >= 1"
    exit 1
fi

if [[ ! -x "$BINARY" ]]; then
    echo "Error: binary not found or not executable: $BINARY"
    exit 1
fi

if [[ "$RESET_BEFORE_EACH" -eq 1 ]] && ! command -v tt-smi >/dev/null 2>&1; then
    echo "Error: --reset/--reset-before-each requested but tt-smi is not available in PATH"
    exit 1
fi

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

TT_METAL_HOME_DEFAULT="/scratch/javier/tt-metal"
if [[ -z "${TT_METAL_HOME:-}" ]]; then
    export TT_METAL_HOME="$TT_METAL_HOME_DEFAULT"
fi
if [[ -z "${TT_METAL_RUNTIME_ROOT:-}" ]]; then
    export TT_METAL_RUNTIME_ROOT="$TT_METAL_HOME"
fi

if [[ -z "$CACHE_DIR" ]]; then
    if [[ -n "${TT_METAL_CACHE:-}" ]]; then
        CACHE_DIR="$TT_METAL_CACHE"
    elif [[ -n "${HOME:-}" ]]; then
        CACHE_DIR="$HOME/.cache/tt-metal-cache"
    else
        CACHE_DIR="/tmp/tt-metal-cache"
    fi
fi
export TT_METAL_CACHE="$CACHE_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -z "$OUT_DIR" ]]; then
    CACHE_TAG="$(safe_tag "$CACHE_MODE")"
    if [[ "$MODE" == "phases" ]]; then
        PHASES_TAG="$(safe_tag "$PHASES")"
        OUT_BASE="pgm_dispatch_suite_m${MODE}_ph${PHASES_TAG}_w${WARMUP}_i${ITERATIONS}_r${REPEATS}_c${CACHE_TAG}"
    else
        OUT_BASE="pgm_dispatch_suite_m${MODE}_w${WARMUP}_i${ITERATIONS}_r${REPEATS}_c${CACHE_TAG}"
    fi
    if [[ "$NO_TIMESTAMP" -eq 1 ]]; then
        OUT_DIR="logs/${OUT_BASE}"
    else
        OUT_DIR="logs/${OUT_BASE}_${STAMP}"
    fi
fi

if [[ -z "$LOG_DIR" ]]; then
    LOG_DIR="logs/pgm_dispatch_logs_${MODE}"
fi

mkdir -p "$OUT_DIR"
mkdir -p "$LOG_DIR"

if [[ "$DRY_RUN" -eq 0 ]]; then
    # Keep summary outputs historical, but always overwrite raw run logs.
    find "$LOG_DIR" -maxdepth 1 -type f -name '*.log' -delete
    if [[ "$CACHE_MODE" == "clean-start" ]]; then
        echo "Cleaning cache directory once at start: $CACHE_DIR"
        rm -rf "$CACHE_DIR"
        mkdir -p "$CACHE_DIR"
    fi
fi

SUMMARY_TSV="$OUT_DIR/summary.tsv"
MEASUREMENTS_TSV="$OUT_DIR/measurements.tsv"
SUMMARY_CSV="$OUT_DIR/summary.csv"
MEASUREMENTS_CSV="$OUT_DIR/measurements.csv"
MATRIX32_PIVOT_CSV="$OUT_DIR/matrix32_pivot.csv"

printf "idx\tphase\tname\tgrid\truntime_args\ttrace\tfinish_only\tcache_mode\tcache_dir\trepeats\tstatus\texit_code\tduration_sec_total\ttotal_us_mean\ttotal_us_std\tper_iter_us_mean\tper_iter_us_std\tlog_prefix\tcommand\n" > "$SUMMARY_TSV"
printf "idx\tphase\tname\trep\tgrid\truntime_args\ttrace\tfinish_only\tcache_mode\tcache_dir\tstatus\texit_code\tduration_sec\ttotal_us\tper_iter_us\tlog_file\tcommand\n" > "$MEASUREMENTS_TSV"

declare -a CASE_PHASES
declare -a CASE_NAMES
declare -a CASE_CMDS
declare -a CASE_GRIDS
declare -a CASE_RT_ARGS
declare -a CASE_TRACE
declare -a CASE_FINISH

has_phase() {
    local phase="$1"
    [[ ",${PHASES}," == *",${phase},"* ]]
}

add_custom_case() {
    local phase="$1"
    local name="$2"
    local args="$3"
    local grid="${4:-NA}"
    local runtime_args="${5:-NA}"
    local trace="${6:-NA}"
    local finish_only="${7:-NA}"
    local cmd
    cmd="$BINARY --custom -w $WARMUP -i $ITERATIONS $args"
    CASE_PHASES+=("$phase")
    CASE_NAMES+=("$name")
    CASE_CMDS+=("$cmd")
    CASE_GRIDS+=("$grid")
    CASE_RT_ARGS+=("$runtime_args")
    CASE_TRACE+=("$trace")
    CASE_FINISH+=("$finish_only")
}

safe_name() {
    echo "$1" | tr ' /' '__' | tr -cd 'A-Za-z0-9._-'
}

if [[ "$MODE" == "phases" ]]; then
    if has_phase "A"; then
        add_custom_case "A" "A_min_brisc_args8" "-x 0 -y 0 -a 8 -ca 0 -c 0 -S 0 -n -t"
        add_custom_case "A" "A_min_brisc_args8_finish_only" "-x 0 -y 0 -a 8 -ca 0 -c 0 -S 0 -n -t -f"
        add_custom_case "A" "A_min_brisc_args8_trace" "-x 0 -y 0 -a 8 -ca 0 -c 0 -S 0 -n -t -tr"
        add_custom_case "A" "A_min_brisc_args8_trace_finish_only" "-x 0 -y 0 -a 8 -ca 0 -c 0 -S 0 -n -t -tr -f"
    fi

    if has_phase "B"; then
        for arg_count in 8 32 128 255; do
            add_custom_case "B" "B_args${arg_count}_nontrace" "-x 0 -y 0 -a ${arg_count} -ca 0 -c 0 -S 0 -n -t"
            add_custom_case "B" "B_args${arg_count}_nontrace_finish_only" "-x 0 -y 0 -a ${arg_count} -ca 0 -c 0 -S 0 -n -t -f"
            add_custom_case "B" "B_args${arg_count}_trace" "-x 0 -y 0 -a ${arg_count} -ca 0 -c 0 -S 0 -n -t -tr"
            add_custom_case "B" "B_args${arg_count}_trace_finish_only" "-x 0 -y 0 -a ${arg_count} -ca 0 -c 0 -S 0 -n -t -tr -f"
        done
    fi

    if has_phase "C"; then
        for arg_count in 8 255; do
            add_custom_case "C" "C_realistic_a${arg_count}_base_nontrace" "-x 6 -y 6 -a ${arg_count} -ca 0 -c 0 -S 0"
            add_custom_case "C" "C_realistic_a${arg_count}_base_nontrace_finish_only" "-x 6 -y 6 -a ${arg_count} -ca 0 -c 0 -S 0 -f"
            add_custom_case "C" "C_realistic_a${arg_count}_base_trace" "-x 6 -y 6 -a ${arg_count} -ca 0 -c 0 -S 0 -tr"
            add_custom_case "C" "C_realistic_a${arg_count}_base_trace_finish_only" "-x 6 -y 6 -a ${arg_count} -ca 0 -c 0 -S 0 -tr -f"

            add_custom_case "C" "C_realistic_a${arg_count}_midmeta_nontrace" "-x 6 -y 6 -a ${arg_count} -ca 0 -c 8 -S 1"
            add_custom_case "C" "C_realistic_a${arg_count}_midmeta_nontrace_finish_only" "-x 6 -y 6 -a ${arg_count} -ca 0 -c 8 -S 1 -f"
            add_custom_case "C" "C_realistic_a${arg_count}_midmeta_trace" "-x 6 -y 6 -a ${arg_count} -ca 0 -c 8 -S 1 -tr"
            add_custom_case "C" "C_realistic_a${arg_count}_midmeta_trace_finish_only" "-x 6 -y 6 -a ${arg_count} -ca 0 -c 8 -S 1 -tr -f"

            add_custom_case "C" "C_realistic_a${arg_count}_highmeta_nontrace" "-x 6 -y 6 -a ${arg_count} -ca 0 -c 32 -S 4"
            add_custom_case "C" "C_realistic_a${arg_count}_highmeta_nontrace_finish_only" "-x 6 -y 6 -a ${arg_count} -ca 0 -c 32 -S 4 -f"
            add_custom_case "C" "C_realistic_a${arg_count}_highmeta_trace" "-x 6 -y 6 -a ${arg_count} -ca 0 -c 32 -S 4 -tr"
            add_custom_case "C" "C_realistic_a${arg_count}_highmeta_trace_finish_only" "-x 6 -y 6 -a ${arg_count} -ca 0 -c 32 -S 4 -tr -f"
        done
    fi

    if has_phase "D"; then
        for kg in 1 4 7; do
            for sd in 1 2 4; do
                add_custom_case "D" "D_kg${kg}_sd${sd}_nontrace" "-x 6 -y 6 -a 255 -ca 0 -c 8 -S 1 -kg ${kg} -sd ${sd}"
                add_custom_case "D" "D_kg${kg}_sd${sd}_nontrace_finish_only" "-x 6 -y 6 -a 255 -ca 0 -c 8 -S 1 -kg ${kg} -sd ${sd} -f"
                add_custom_case "D" "D_kg${kg}_sd${sd}_trace" "-x 6 -y 6 -a 255 -ca 0 -c 8 -S 1 -kg ${kg} -sd ${sd} -tr"
                add_custom_case "D" "D_kg${kg}_sd${sd}_trace_finish_only" "-x 6 -y 6 -a 255 -ca 0 -c 8 -S 1 -kg ${kg} -sd ${sd} -tr -f"
            done
        done
    fi
else
    for grid_label in single_1x1 full_7x7; do
        if [[ "$grid_label" == "single_1x1" ]]; then
            grid_args="-x 0 -y 0"
            grid_text="1x1"
        else
            grid_args="-x 6 -y 6"
            grid_text="7x7"
        fi
        for arg_count in 8 64 128 255; do
            for trace_flag in 0 1; do
                for finish_flag in 0 1; do
                    cfg_args="$grid_args -a ${arg_count} -ca 0 -c 0 -S 0"
                    if [[ "$trace_flag" -eq 1 ]]; then
                        cfg_args+=" -tr"
                    fi
                    if [[ "$finish_flag" -eq 1 ]]; then
                        cfg_args+=" -f"
                    fi
                    add_custom_case \
                        "M32" \
                        "M32_${grid_label}_a${arg_count}_tr${trace_flag}_f${finish_flag}" \
                        "$cfg_args" \
                        "$grid_text" \
                        "$arg_count" \
                        "$trace_flag" \
                        "$finish_flag"
                done
            done
        done
    done
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

echo "Suite output directory: $OUT_DIR"
echo "Log directory (overwritten): $LOG_DIR"
if [[ "$MODE" == "phases" ]]; then
    echo "Selected phases: $PHASES"
else
    echo "Selected mode: matrix32"
fi
echo "Total custom cases: ${#CASE_NAMES[@]}"
echo "Repeats per case: $REPEATS"
echo "Cache mode: $CACHE_MODE"
echo "Cache directory: $CACHE_DIR"
if [[ -n "$EXEC_PREFIX" ]]; then
    echo "Execution prefix: $EXEC_PREFIX"
fi

if [[ ${#CASE_NAMES[@]} -eq 0 ]]; then
    echo "Error: no cases selected. Check --phases argument."
    exit 1
fi

fail_count=0
pass_count=0
skip_count=0

for idx in "${!CASE_NAMES[@]}"; do
    phase="${CASE_PHASES[$idx]}"
    name="${CASE_NAMES[$idx]}"
    cmd_raw="${CASE_CMDS[$idx]}"
    cmd="${EXEC_PREFIX}${cmd_raw}"
    grid="${CASE_GRIDS[$idx]}"
    runtime_args="${CASE_RT_ARGS[$idx]}"
    trace_mode="${CASE_TRACE[$idx]}"
    finish_only="${CASE_FINISH[$idx]}"
    log_prefix="$LOG_DIR/$(printf "%03d" "$idx")_$(safe_name "$name")"
    case_metrics_file="$OUT_DIR/$(printf "%03d" "$idx")_$(safe_name "$name").metrics.tsv"
    : > "$case_metrics_file"

    echo
    echo "[$((idx + 1))/${#CASE_NAMES[@]}] phase=$phase name=$name"
    echo "Command: $cmd"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        status="SKIPPED_DRY_RUN"
        rc=0
        duration_sec_total=0
        total_us_mean="NA"
        total_us_std="NA"
        per_iter_us_mean="NA"
        per_iter_us_std="NA"
        skip_count=$((skip_count + 1))
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$idx" "$phase" "$name" "1" "$grid" "$runtime_args" "$trace_mode" "$finish_only" \
            "$CACHE_MODE" "$CACHE_DIR" "$status" "$rc" "0" "NA" "NA" "${log_prefix}_r01.log" "$cmd" >> "$MEASUREMENTS_TSV"
    else
        status="PASS"
        rc=0
        duration_sec_total=0
        for rep in $(seq 1 "$REPEATS"); do
            log_file="${log_prefix}_r$(printf "%02d" "$rep").log"
            if [[ "$CACHE_MODE" == "clean-case" ]]; then
                rm -rf "$CACHE_DIR"
                mkdir -p "$CACHE_DIR"
            fi
            if [[ "$RESET_BEFORE_EACH" -eq 1 ]]; then
                tt-smi -r 0 > "${log_prefix}_r$(printf "%02d" "$rep")_reset.log" 2>&1 || true
            fi

            start_sec=$(date +%s)
            bash -lc "$cmd" |& tee "$log_file"
            rc_rep=${PIPESTATUS[0]}
            end_sec=$(date +%s)
            duration_rep=$((end_sec - start_sec))
            duration_sec_total=$((duration_sec_total + duration_rep))

            total_us="NA"
            per_iter_us="NA"
            if [[ $rc_rep -eq 0 ]]; then
                total_us="$(extract_total_us "$log_file")"
                per_iter_us="$(extract_per_iter_us "$log_file")"
            fi

            rep_status="PASS"
            if [[ $rc_rep -ne 0 ]]; then
                rep_status="FAIL"
                status="FAIL"
                rc=1
            fi

            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                "$idx" "$phase" "$name" "$rep" "$grid" "$runtime_args" "$trace_mode" "$finish_only" \
                "$CACHE_MODE" "$CACHE_DIR" "$rep_status" "$rc_rep" "$duration_rep" "$total_us" "$per_iter_us" "$log_file" "$cmd" >> "$MEASUREMENTS_TSV"

            printf "%s\t%s\t%s\t%s\n" "$rep" "$total_us" "$per_iter_us" "$rc_rep" >> "$case_metrics_file"
        done

        read -r total_us_mean total_us_std < <(compute_mean_std "$case_metrics_file" 2)
        read -r per_iter_us_mean per_iter_us_std < <(compute_mean_std "$case_metrics_file" 3)

        if [[ "$status" == "PASS" ]]; then
            pass_count=$((pass_count + 1))
        else
            fail_count=$((fail_count + 1))
        fi
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$idx" "$phase" "$name" "$grid" "$runtime_args" "$trace_mode" "$finish_only" "$CACHE_MODE" "$CACHE_DIR" "$REPEATS" "$status" "$rc" \
        "$duration_sec_total" "$total_us_mean" "$total_us_std" "$per_iter_us_mean" "$per_iter_us_std" "$log_prefix" "$cmd" >> "$SUMMARY_TSV"
done

if [[ "$RUN_GBENCH" -eq 1 ]]; then
    gbench_log="$OUT_DIR/gbench.log"
    gbench_json="$OUT_DIR/gbench.json"
    gbench_cmd="$BINARY --benchmark_filter=\"$GBENCH_FILTER\" --benchmark_out=\"$gbench_json\" --benchmark_out_format=json"

    echo
    echo "Running optional Google Benchmark mode"
    echo "Command: $gbench_cmd"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "SKIPPED_DRY_RUN: Google Benchmark mode"
    else
        bash -lc "$gbench_cmd" |& tee "$gbench_log"
    fi
fi

echo
echo "Summary TSV: $SUMMARY_TSV"
cat "$SUMMARY_TSV"

echo
echo "Measurements TSV: $MEASUREMENTS_TSV"

tsv_to_csv "$SUMMARY_TSV" "$SUMMARY_CSV"
tsv_to_csv "$MEASUREMENTS_TSV" "$MEASUREMENTS_CSV"

echo
echo "Summary CSV: $SUMMARY_CSV"
echo "Measurements CSV: $MEASUREMENTS_CSV"

if [[ "$MODE" == "matrix32" ]]; then
    generate_matrix32_pivot_csv "$SUMMARY_TSV" "$MATRIX32_PIVOT_CSV"
    echo "Matrix32 Pivot CSV: $MATRIX32_PIVOT_CSV"
fi

echo
echo "Counts: pass=$pass_count fail=$fail_count skipped=$skip_count"

if [[ "$DRY_RUN" -eq 1 ]]; then
    exit 0
fi

if [[ $fail_count -gt 0 ]]; then
    exit 1
fi

exit 0
