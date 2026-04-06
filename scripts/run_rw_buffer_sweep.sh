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

BINARY="build/test_rw_buffer"
DEVICE_ID=0
NUM_TESTS=20
REPEATS=3
BUFFER_TYPES="0"
TRANSFER_SIZES="67108864,268435456,536870912,1073741824"
PAGE_SIZES="2048,8192,32768"
DO_READ=1
DO_WRITE=1
BYPASS_CHECK=1
DRY_RUN=0
OUT_DIR=""
LOG_DIR=""
TASKSET_CPUS=""
NUMACTL_CPUBIND=""
NUMACTL_MEMBIND=""
NO_TIMESTAMP=0
RESET_DEVICE=0

TT_METAL_HOME_DEFAULT="$ROOT_DIR"

usage() {
    cat <<'EOF'
Usage:
  ./run_rw_buffer_sweep.sh [options]

Options:
    --binary PATH              Path to test_rw_buffer binary (default: build/test_rw_buffer)
  --device ID                Device id (default: 0)
  --num-tests N              Internal test iterations in test_rw_buffer (default: 20)
  --repeats N                Repeat each sweep point N times (default: 3)
  --buffer-types LIST        Comma list: 0 (DRAM), 1 (L1) (default: 0)
  --transfer-sizes LIST      Comma list in bytes (default: 64M,256M,512M,1G)
  --page-sizes LIST          Comma list in bytes (default: 2048,8192,32768)
    --mode MODE                full|write-only|read-only (default: full)
    --no-bypass-check          Disable bypass-check (default keeps bypass enabled)
    --out-dir DIR              Output directory path
  --log-dir DIR              Log directory path (cleaned each non-dry run)
    --reset                    Reset device with tt-smi -r 0 before each command
  --no-timestamp             Use deterministic output dir when --out-dir is not set
  --taskset-cpus LIST        CPU pinning range/list for host process
  --numactl-cpubind NODE     NUMA CPU binding node
  --numactl-membind NODE     NUMA memory binding node
  --dry-run                  Print commands only
  -h, --help                 Show this help

Outputs:
  summary.tsv / summary.csv
  measurements.tsv / measurements.csv

Notes:
  - Logs are overwritten each run in log directory.
    - If --out-dir is not provided, output defaults to <repo-root>/logs/rw_buffer_sweep_<config>_<timestamp>.
  - Use DRAM (buffer-type 0) for realistic large ML transfer studies.
EOF
}

is_uint() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

safe_tag() {
    echo "$1" | tr ', ' '__' | tr -cd 'A-Za-z0-9._-'
}

extract_csv_h2d() {
    local log_file="$1"
    awk '
        match($0,/CSV_OUTPUT:H2D_Bandwidth\(GB\/s\):([0-9]+(\.[0-9]+)?):D2H_Bandwidth\(GB\/s\):([0-9]+(\.[0-9]+)?)/,m){v=m[1]}
        END{if(v=="") print "NA"; else print v}
    ' "$log_file"
}

extract_csv_d2h() {
    local log_file="$1"
    awk '
        match($0,/CSV_OUTPUT:H2D_Bandwidth\(GB\/s\):([0-9]+(\.[0-9]+)?):D2H_Bandwidth\(GB\/s\):([0-9]+(\.[0-9]+)?)/,m){v=m[3]}
        END{if(v=="") print "NA"; else print v}
    ' "$log_file"
}

extract_best_write() {
    local log_file="$1"
    awk 'match($0,/Best write: ([0-9]+(\.[0-9]+)?) GB\/s/,m){v=m[1]} END{if(v=="") print "NA"; else print v}' "$log_file"
}

extract_best_read() {
    local log_file="$1"
    awk 'match($0,/Best read: ([0-9]+(\.[0-9]+)?) GB\/s/,m){v=m[1]} END{if(v=="") print "NA"; else print v}' "$log_file"
}

extract_csv_pass() {
    local log_file="$1"
    awk '
        match($0,/CSV_RESULT:pass:(true|false)/,m){v=m[1]}
        END{if(v=="") print "unknown"; else print v}
    ' "$log_file"
}

compute_mean_std() {
    local metrics_file="$1"
    local field_index="$2"
    awk -F'\t' -v idx="$field_index" '
        $idx != "NA" && $idx != "" {n++; s+=$idx; ss+=$idx*$idx}
        END {
            if (n == 0) { print "NA\tNA"; exit }
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

MODE="full"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --binary)
            BINARY="$2"
            shift 2
            ;;
        --device)
            DEVICE_ID="$2"
            shift 2
            ;;
        --num-tests)
            NUM_TESTS="$2"
            shift 2
            ;;
        --repeats)
            REPEATS="$2"
            shift 2
            ;;
        --buffer-types)
            BUFFER_TYPES="$2"
            shift 2
            ;;
        --transfer-sizes)
            TRANSFER_SIZES="$2"
            shift 2
            ;;
        --page-sizes)
            PAGE_SIZES="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --no-bypass-check)
            BYPASS_CHECK=0
            shift
            ;;
        --out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --reset)
            RESET_DEVICE=1
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

if [[ "$MODE" == "full" ]]; then
    DO_READ=1
    DO_WRITE=1
elif [[ "$MODE" == "write-only" ]]; then
    DO_READ=0
    DO_WRITE=1
elif [[ "$MODE" == "read-only" ]]; then
    DO_READ=1
    DO_WRITE=0
else
    echo "Error: --mode must be full|write-only|read-only"
    exit 1
fi

BINARY="$(to_abs_from_root "$BINARY")"
if [[ -n "$OUT_DIR" ]]; then
    OUT_DIR="$(to_abs_from_root "$OUT_DIR")"
fi
if [[ -n "$LOG_DIR" ]]; then
    LOG_DIR="$(to_abs_from_root "$LOG_DIR")"
fi

if ! is_uint "$DEVICE_ID" || ! is_uint "$NUM_TESTS" || ! is_uint "$REPEATS"; then
    echo "Error: --device, --num-tests and --repeats must be integers"
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

if [[ -n "$TASKSET_CPUS" ]] && ! command -v taskset >/dev/null 2>&1; then
    echo "Error: taskset is not available"
    exit 1
fi
if [[ -n "$NUMACTL_CPUBIND" || -n "$NUMACTL_MEMBIND" ]]; then
    if ! command -v numactl >/dev/null 2>&1; then
        echo "Error: numactl is not available"
        exit 1
    fi
fi
if [[ "$RESET_DEVICE" -eq 1 ]] && ! command -v tt-smi >/dev/null 2>&1; then
    echo "Error: --reset requested but tt-smi is not available"
    exit 1
fi

if [[ -z "${TT_METAL_HOME:-}" ]]; then
    export TT_METAL_HOME="$TT_METAL_HOME_DEFAULT"
fi
if [[ -z "${TT_METAL_RUNTIME_ROOT:-}" ]]; then
    export TT_METAL_RUNTIME_ROOT="$TT_METAL_HOME"
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -z "$OUT_DIR" ]]; then
    MODE_TAG="$(safe_tag "$MODE")"
    BUFFER_TAG="$(safe_tag "$BUFFER_TYPES")"
    XFER_TAG="$(safe_tag "$TRANSFER_SIZES")"
    PAGE_TAG="$(safe_tag "$PAGE_SIZES")"
    OUT_BASE="rw_buffer_sweep_m${MODE_TAG}_b${BUFFER_TAG}_x${XFER_TAG}_p${PAGE_TAG}_n${NUM_TESTS}_r${REPEATS}"
    if [[ "$NO_TIMESTAMP" -eq 1 ]]; then
        OUT_DIR="$ROOT_DIR/logs/${OUT_BASE}"
    else
        OUT_DIR="$ROOT_DIR/logs/${OUT_BASE}_${STAMP}"
    fi
fi
if [[ -z "$LOG_DIR" ]]; then
    LOG_DIR="$ROOT_DIR/logs/rw_buffer_logs"
fi
mkdir -p "$OUT_DIR"
mkdir -p "$LOG_DIR"

if [[ "$DRY_RUN" -eq 0 ]]; then
    find "$LOG_DIR" -maxdepth 1 -type f -name '*.log' -delete
fi

SUMMARY_TSV="$OUT_DIR/summary.tsv"
MEASUREMENTS_TSV="$OUT_DIR/measurements.tsv"
SUMMARY_CSV="$OUT_DIR/summary.csv"
MEASUREMENTS_CSV="$OUT_DIR/measurements.csv"

printf "idx\tbuffertype\ttransfer_size\tpage_size\tmode\trepeats\tstatus\texit_code\th2d_mean\th2d_std\td2h_mean\td2h_std\tbest_write_mean\tbest_write_std\tbest_read_mean\tbest_read_std\tlog_prefix\tcommand\n" > "$SUMMARY_TSV"
printf "idx\trep\tbuffertype\ttransfer_size\tpage_size\tmode\tstatus\texit_code\th2d\td2h\tbest_write\tbest_read\tlog_file\tcommand\n" > "$MEASUREMENTS_TSV"

IFS=',' read -r -a BUFFER_LIST <<< "$BUFFER_TYPES"
IFS=',' read -r -a XFER_LIST <<< "$TRANSFER_SIZES"
IFS=',' read -r -a PAGE_LIST <<< "$PAGE_SIZES"

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

idx=0
pass_count=0
fail_count=0
skip_count=0

for buf in "${BUFFER_LIST[@]}"; do
    for xfer in "${XFER_LIST[@]}"; do
        for page in "${PAGE_LIST[@]}"; do
            name="rw_b${buf}_x${xfer}_p${page}_${MODE}"
            log_prefix="$LOG_DIR/$(printf "%03d" "$idx")_${name}"
            metric_file="$OUT_DIR/$(printf "%03d" "$idx")_${name}.metrics.tsv"
            : > "$metric_file"

            base_cmd="$BINARY --device ${DEVICE_ID} --buffer-type ${buf} --transfer-size ${xfer} --page-size ${page} --num-tests ${NUM_TESTS}"
            if [[ "$BYPASS_CHECK" -eq 1 ]]; then
                base_cmd+=" --bypass-check"
            fi
            if [[ "$DO_READ" -eq 0 ]]; then
                base_cmd+=" --skip-read"
            fi
            if [[ "$DO_WRITE" -eq 0 ]]; then
                base_cmd+=" --skip-write"
            fi
            cmd="${EXEC_PREFIX}${base_cmd}"

            echo
            echo "[$((idx + 1))] $name"
            echo "Command: $cmd"

            if [[ "$DRY_RUN" -eq 1 ]]; then
                status="SKIPPED_DRY_RUN"
                rc=0
                skip_count=$((skip_count + 1))
                printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                    "$idx" "1" "$buf" "$xfer" "$page" "$MODE" "$status" "$rc" "NA" "NA" "NA" "NA" "${log_prefix}_r01.log" "$cmd" >> "$MEASUREMENTS_TSV"
                h2d_mean="NA"; h2d_std="NA"; d2h_mean="NA"; d2h_std="NA"
                bw_mean="NA"; bw_std="NA"; br_mean="NA"; br_std="NA"
            else
                status="PASS"
                rc=0
                for rep in $(seq 1 "$REPEATS"); do
                    log_file="${log_prefix}_r$(printf "%02d" "$rep").log"
                    if [[ "$RESET_DEVICE" -eq 1 ]]; then
                        tt-smi -r 0 > "${log_prefix}_r$(printf "%02d" "$rep")_reset.log" 2>&1 || true
                    fi
                    bash -lc "$cmd" |& tee "$log_file"
                    rc_rep=${PIPESTATUS[0]}

                    h2d="NA"; d2h="NA"; bestw="NA"; bestr="NA"
                    pass_flag="unknown"
                    if [[ $rc_rep -eq 0 ]]; then
                        h2d="$(extract_csv_h2d "$log_file")"
                        d2h="$(extract_csv_d2h "$log_file")"
                        bestw="$(extract_best_write "$log_file")"
                        bestr="$(extract_best_read "$log_file")"
                        pass_flag="$(extract_csv_pass "$log_file")"
                    fi

                    rep_status="PASS"
                    if [[ $rc_rep -ne 0 || "$pass_flag" == "false" ]]; then
                        rep_status="FAIL"
                        status="FAIL"
                        rc=1
                    fi

                    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                        "$idx" "$rep" "$buf" "$xfer" "$page" "$MODE" "$rep_status" "$rc_rep" "$h2d" "$d2h" "$bestw" "$bestr" "$log_file" "$cmd" >> "$MEASUREMENTS_TSV"
                    printf "%s\t%s\t%s\t%s\t%s\n" "$rep" "$h2d" "$d2h" "$bestw" "$bestr" >> "$metric_file"
                done

                read -r h2d_mean h2d_std < <(compute_mean_std "$metric_file" 2)
                read -r d2h_mean d2h_std < <(compute_mean_std "$metric_file" 3)
                read -r bw_mean bw_std < <(compute_mean_std "$metric_file" 4)
                read -r br_mean br_std < <(compute_mean_std "$metric_file" 5)

                if [[ "$status" == "PASS" ]]; then
                    pass_count=$((pass_count + 1))
                else
                    fail_count=$((fail_count + 1))
                fi
            fi

            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                "$idx" "$buf" "$xfer" "$page" "$MODE" "$REPEATS" "$status" "$rc" \
                "$h2d_mean" "$h2d_std" "$d2h_mean" "$d2h_std" "$bw_mean" "$bw_std" "$br_mean" "$br_std" "$log_prefix" "$cmd" >> "$SUMMARY_TSV"

            idx=$((idx + 1))
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
echo "Log directory (overwritten): $LOG_DIR"
echo "Counts: pass=$pass_count fail=$fail_count skipped=$skip_count"

if [[ "$DRY_RUN" -eq 1 ]]; then
    exit 0
fi
if [[ $fail_count -gt 0 ]]; then
    exit 1
fi
exit 0
