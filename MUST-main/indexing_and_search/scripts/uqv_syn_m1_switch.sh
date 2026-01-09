#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

DATASET_ROOT=${DATASET_ROOT:-$REPO_ROOT}
BUILD_TYPE=${BUILD_TYPE:-release}
STRATEGY=${STRATEGY:-MSAG}
TOPK_LIST=${TOPK_LIST:-"10"}
LAMBDA=${LAMBDA:-0.5}
LAMBDA_LIST=${LAMBDA_LIST:-"0.1 0.3 0.5 0.7 0.9"}
WEIGHT_SIGN=${WEIGHT_SIGN:--1}
SEARCH_L_OVERRIDE=${SEARCH_L_OVERRIDE:-}

SUB_DATASET_BASE=${SUB_DATASET_BASE:-s_5}
AUX_A_SET=${AUX_A_SET:-s_1}
AUX_B_SET=${AUX_B_SET:-s_10}

scale_weight() {
  awk -v s="${WEIGHT_SIGN}" -v v="$1" 'BEGIN {printf "%.6f", s * v}'
}

complement_lambda() {
  awk -v v="$1" 'BEGIN {printf "%.6f", 1.0 - v}'
}

S_TAG_BASE=${SUB_DATASET_BASE#s_}
S_TAG_A=${AUX_A_SET#s_}
S_TAG_B=${AUX_B_SET#s_}

BASE_DIR="${DATASET_ROOT}/uqv_syn/${SUB_DATASET_BASE}"
AUX_A_DIR="${DATASET_ROOT}/uqv_syn/${AUX_A_SET}"
AUX_B_DIR="${DATASET_ROOT}/uqv_syn/${AUX_B_SET}"

BASE_MODAL1="${BASE_DIR}/random_base_n100000_d32_c10_s${S_TAG_BASE}.fvecs"
QUERY_MODAL1="${BASE_DIR}/random_query_n1000_d32_c10_s${S_TAG_BASE}.fvecs"
GROUNDTRUTH="${BASE_DIR}/random_ground_truth_n1000_d32_c10_s${S_TAG_BASE}.ivecs"

BASE_MODAL2_A="${AUX_A_DIR}/random_base_n100000_d32_c10_s${S_TAG_A}.fvecs"
QUERY_MODAL2_A="${AUX_A_DIR}/random_query_n1000_d32_c10_s${S_TAG_A}.fvecs"
BASE_MODAL2_B="${AUX_B_DIR}/random_base_n100000_d32_c10_s${S_TAG_B}.fvecs"
QUERY_MODAL2_B="${AUX_B_DIR}/random_query_n1000_d32_c10_s${S_TAG_B}.fvecs"

if [[ ! -f "${BASE_MODAL1}" || ! -f "${QUERY_MODAL1}" || ! -f "${GROUNDTRUTH}" ]]; then
  echo "[SCRIPT] Missing base/query/groundtruth for ${SUB_DATASET_BASE}" >&2
  exit 1
fi

if [[ ! -f "${BASE_MODAL2_A}" || ! -f "${QUERY_MODAL2_A}" ]]; then
  echo "[SCRIPT] Missing auxiliary set ${AUX_A_SET}" >&2
  exit 1
fi

if [[ ! -f "${BASE_MODAL2_B}" || ! -f "${QUERY_MODAL2_B}" ]]; then
  echo "[SCRIPT] Missing auxiliary set ${AUX_B_SET}" >&2
  exit 1
fi

export DATASET_ROOT
export DATASET=uqv_syn
export SUB_DATASET="${SUB_DATASET_BASE}"
export STRATEGY
export BUILD_THREAD_NUM=${BUILD_THREAD_NUM:-1}

RUN_TAG=$(date +%Y_%m_%d_%H%M%S)
RESULTS_FILE="${SCRIPT_DIR}/../doc/result/uqv_syn_m1_switch_${RUN_TAG}.tsv"
echo -e "group\tlambda\ttopk\trecall" > "${RESULTS_FILE}"

INDEX_PATH="${SCRIPT_DIR}/../doc/index/${DATASET}/${DATASET}_${STRATEGY}.index"
if [[ -f "${INDEX_PATH}" && "${FORCE_BUILD:-0}" != "1" ]]; then
  echo "[SCRIPT] Index exists at ${INDEX_PATH}; skip build (FORCE_BUILD=1 to rebuild)."
else
  echo "[SCRIPT] Build index on ${SUB_DATASET_BASE} (fixed graph)"
  EXTRA_CMAKE_ARGS=${EXTRA_CMAKE_ARGS:--DMODAL2_FLOAT=1} \
    "${SCRIPT_DIR}/run.sh" "${BUILD_TYPE}" "build_${STRATEGY}"
fi

run_group() {
  local label=$1
  local base2=$2
  local query2=$3

  local find_last_eva
  if command -v rg >/dev/null 2>&1; then
    find_last_eva='rg -n "\\[EVA\\]"'
  else
    find_last_eva='grep -n "\\[EVA\\]"'
  fi

  for lambda in ${LAMBDA_LIST}; do
    local w1_cur
    local w2_cur
    w1_cur=$(scale_weight "${lambda}")
    w2_cur=$(scale_weight "$(complement_lambda "${lambda}")")
    for k in ${TOPK_LIST}; do
      echo "[SCRIPT] Group ${label} | lambda=${lambda} | R@${k}"
      SEARCH_MODAL1_BASE_PATH="${BASE_MODAL1}" \
      SEARCH_MODAL2_BASE_PATH="${base2}" \
      SEARCH_MODAL1_QUERY_PATH="${QUERY_MODAL1}" \
      SEARCH_MODAL2_QUERY_PATH="${query2}" \
      SEARCH_GROUNDTRUTH_PATH="${GROUNDTRUTH}" \
      SEARCH_W1="${w1_cur}" SEARCH_W2="${w2_cur}" \
      SEARCH_TOPK="${k}" SEARCH_GTK="${k}" \
      SEARCH_L="${SEARCH_L_OVERRIDE}" \
      EXTRA_CMAKE_ARGS=${EXTRA_CMAKE_ARGS:--DMODAL2_FLOAT=1} \
      "${SCRIPT_DIR}/run.sh" "${BUILD_TYPE}" "search_${STRATEGY}"

      log_file=$(ls -t "${SCRIPT_DIR}/../doc/result"/*/search_${STRATEGY}.log 2>/dev/null | head -1 || true)
      if [[ -n "${log_file}" ]]; then
        eva_line=$(eval "${find_last_eva} \"${log_file}\"" | tail -n 1 | sed -E 's/^[0-9]+://')
        if [[ -n "${eva_line}" ]]; then
          recall=$(echo "${eva_line}" | awk -F': ' '{print $2}')
          echo -e "${label}\t${lambda}\t${k}\t${recall}" >> "${RESULTS_FILE}"
          echo "[SUMMARY] ${label} lambda=${lambda} R@${k}=${recall}"
        else
          echo "[SUMMARY] ${label} lambda=${lambda} R@${k}=N/A (no EVA line found)"
        fi
      else
        echo "[SUMMARY] ${label} lambda=${lambda} R@${k}=N/A (log not found)"
      fi
    done
  done
}

run_group "A(${AUX_A_SET})" "${BASE_MODAL2_A}" "${QUERY_MODAL2_A}"
run_group "B(${AUX_B_SET})" "${BASE_MODAL2_B}" "${QUERY_MODAL2_B}"

python "${SCRIPT_DIR}/plot_uqv_syn_results.py" \
  --input "${RESULTS_FILE}" \
  --output-prefix "${RESULTS_FILE%.tsv}" || true
