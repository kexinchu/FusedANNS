#!/bin/bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "[SCRIPT] Usage: $0 <group_a_dir> <group_b_dir>"
  echo "[SCRIPT] Required env: DATASET (and optional DATASET_SUFFIX/TYPE/SUB_DATASET)"
  exit 1
fi

GROUP_A_DIR=$1
GROUP_B_DIR=$2
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [[ -z "${DATASET:-}" ]]; then
  echo "[SCRIPT] DATASET is required to pick search parameters."
  exit 1
fi

BUILD_TYPE=${BUILD_TYPE:-release}
STRATEGY=${STRATEGY:-MSAG}
TOPK_LIST=${TOPK_LIST:-"10 50"}
WEIGHT_SIGN=${WEIGHT_SIGN:--1}
SEARCH_L_OVERRIDE=${SEARCH_L_OVERRIDE:-}
INDEX_PATH_OVERRIDE=${INDEX_PATH_OVERRIDE:-}

find_query2() {
  local dir=$1
  if [[ -f "${dir}/query_modal2.ivecs" ]]; then
    echo "${dir}/query_modal2.ivecs"
    return 0
  fi
  if [[ -f "${dir}/query_modal2.fvecs" ]]; then
    echo "${dir}/query_modal2.fvecs"
    return 0
  fi
  echo "[SCRIPT] Missing query_modal2 file under ${dir}" >&2
  exit 1
}

scale_weight() {
  awk -v s="${WEIGHT_SIGN}" -v v="$1" 'BEGIN {printf "%.6f", s * v}'
}

declare -a WEIGHT_PAIRS=(
  "0.9 0.1"
  "0.1 0.9"
  "0.5 0.5"
)

run_group() {
  local group_name=$1
  local group_dir=$2
  local q1="${group_dir}/query_modal1.fvecs"
  local q2
  q2=$(find_query2 "${group_dir}")
  local gt="${group_dir}/groundtruth.ivecs"

  if [[ ! -f "${q1}" || ! -f "${gt}" ]]; then
    echo "[SCRIPT] Missing query/groundtruth files in ${group_dir}" >&2
    exit 1
  fi

  for pair in "${WEIGHT_PAIRS[@]}"; do
    read -r lambda_img lambda_text <<< "${pair}"
    local w1
    local w2
    w1=$(scale_weight "${lambda_img}")
    w2=$(scale_weight "${lambda_text}")

    for k in ${TOPK_LIST}; do
      echo "[SCRIPT] Group ${group_name} | lambda_img=${lambda_img} lambda_text=${lambda_text} | R@${k}"
      SEARCH_W1="${w1}" SEARCH_W2="${w2}" SEARCH_TOPK="${k}" SEARCH_GTK="${k}" \
      SEARCH_MODAL1_QUERY_PATH="${q1}" SEARCH_MODAL2_QUERY_PATH="${q2}" \
      SEARCH_GROUNDTRUTH_PATH="${gt}" SEARCH_INDEX_PATH="${INDEX_PATH_OVERRIDE}" \
      SEARCH_L="${SEARCH_L_OVERRIDE}" \
      "${SCRIPT_DIR}/run.sh" "${BUILD_TYPE}" "search_${STRATEGY}"
    done
  done
}

run_group "A" "${GROUP_A_DIR}"
run_group "B" "${GROUP_B_DIR}"
