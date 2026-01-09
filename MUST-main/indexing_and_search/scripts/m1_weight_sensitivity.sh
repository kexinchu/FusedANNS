#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ORIG_DIR=$(pwd)
trap 'cd "${ORIG_DIR}"' EXIT
cd "${SCRIPT_DIR}"
export PARAM_SCRIPT_DIR="${SCRIPT_DIR}"

BUILD_TYPE=${BUILD_TYPE:-release}
STRATEGY=${STRATEGY:-MSAG}
NORM_MODAL2=${NORM_MODAL2:-1}
NORM_MODAL1=${NORM_MODAL1:-}

source "${SCRIPT_DIR}/param.sh"
param_${DATASET}_${STRATEGY}

EXE_PATH="${SCRIPT_DIR}/../${BUILD_TYPE}/S-MSAG"
if [[ ! -x "${EXE_PATH}" ]]; then
  echo "[SCRIPT] Missing ${EXE_PATH}. Build with: ${SCRIPT_DIR}/run.sh ${BUILD_TYPE} search_${STRATEGY}"
  exit 1
fi

RUN_TAG=$(date +%Y_%m_%d_%H%M%S)
OUTPUT_DIR=${OUTPUT_DIR:-"${SCRIPT_DIR}/../doc/result/m1_weight_sensitivity_${RUN_TAG}"}
mkdir -p "${OUTPUT_DIR}"

resolve_scalar() {
  local var_name=$1
  local value
  if declare -p "${var_name}" 2>/dev/null | grep -q 'declare -a'; then
    eval "value=\${${var_name}[0]}"
  else
    eval "value=\${${var_name}}"
  fi
  echo "${value}"
}

TOPK_VAL=${SEARCH_TOPK:-$(resolve_scalar TOPK)}
GTK_VAL=${SEARCH_GTK:-$(resolve_scalar GTK)}
L_SEARCH_VAL=${SEARCH_L_OVERRIDE:-$(resolve_scalar L_search)}

W1_TARGET=${W1_TARGET:-1.0}
W2_TARGET=${W2_TARGET:-0.0}
W1_AUX=${W1_AUX:-0.1}
W2_AUX=${W2_AUX:-2.0}
W1_BALANCE=${W1_BALANCE:-${W1}}
W2_BALANCE=${W2_BALANCE:-${W2}}

SAVE_RESULT_PATH=" "

run_case() {
  local label=$1
  local w1=$2
  local w2=$3
  local out_file="${OUTPUT_DIR}/res_${label}.txt"
  local norm_args=()
  if [[ -n "${NORM_MODAL1}" ]]; then
    norm_args+=(--norm_modal1 "${NORM_MODAL1}")
  fi
  if [[ -n "${NORM_MODAL2}" ]]; then
    norm_args+=(--norm_modal2 "${NORM_MODAL2}")
  fi

  echo "[SCRIPT] ${label}: w1=${w1}, w2=${w2} -> ${out_file}"
  "${EXE_PATH}" \
    "${MODAL1_BASE_PATH}" \
    "${MODAL2_BASE_PATH}" \
    "${MODAL1_QUERY_PATH}" \
    "${MODAL2_QUERY_PATH}" \
    "${GROUNDTRUTH_PATH}" \
    "${INDEX_PATH}" \
    "${SAVE_RESULT_PATH}" \
    "${SEARCH_THREAD_NUM}" \
    "${W1}" \
    "${W2}" \
    "${TOPK_VAL}" \
    "${GTK_VAL}" \
    "${L_SEARCH_VAL}" \
    "${IS_NORM_MODAL1}" \
    "${IS_NORM_MODAL2}" \
    "${IS_SKIP_NUM}" \
    "${SKIP_NUM}" \
    "${IS_MULTI_RESULT_EQUAL}" \
    "${IS_DELETE_ID}" \
    "${DELETE_ID_PATH}" \
    --w1 "${w1}" \
    --w2 "${w2}" \
    --per_query_path "${out_file}" \
    "${norm_args[@]}"
}

run_case "target" "${W1_TARGET}" "${W2_TARGET}"
run_case "aux" "${W1_AUX}" "${W2_AUX}"
run_case "balance" "${W1_BALANCE}" "${W2_BALANCE}"
