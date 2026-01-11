#!/bin/bash

set -e
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ORIG_DIR=$(pwd)
cleanup() {
  cd "$ORIG_DIR"
}
trap cleanup EXIT
cd "$SCRIPT_DIR"

export PARAM_SCRIPT_DIR="$SCRIPT_DIR"
export DATASET=${DATASET:-celeba}
export DATASET_SUFFIX=${DATASET_SUFFIX:-resnet50_encode}
export TYPE=${TYPE:-test}
export STRATEGY=${STRATEGY:-MSAG}

source "${SCRIPT_DIR}/param.sh"
param_${DATASET}_${STRATEGY}

SEARCH_W1=${SEARCH_W1:--0.08}
SEARCH_W2=${SEARCH_W2:--1.18}
ENTRY_TOPK=${ENTRY_TOPK:-1}
THREAD_NUM=${SEARCH_THREAD_NUM:-1}
TOPK_VALUE=${TOPK[0]}
GTK_VALUE=${GTK[0]}
L_SEARCH_VALUE=${L_search[0]}
L_SEARCH_VALUE=${SEARCH_L:-$L_SEARCH_VALUE}

CENTROIDS_VISUAL_PATH=${CENTROIDS_VISUAL_PATH:-"../doc/dataset/celeba_centroids/celeba_centroids_visual.fvecs"}
CENTROIDS_ATTR_PATH=${CENTROIDS_ATTR_PATH:-"../doc/dataset/celeba_centroids/celeba_centroids_attr.ivecs"}

EXE_PATH=${EXE_PATH:-"../release"}
if [ ! -x "${EXE_PATH}/S-MSAG" ]; then
  cmake -DCMAKE_BUILD_TYPE=Release .. -B ../release
  pushd ../release >/dev/null
  make -j
  popd >/dev/null
fi

if [ -z "${RESULT_PREFIX_PATH:-}" ]; then
  TIME=$(date +%Y_%m_%d_%H%M%S)
  RESULT_PREFIX_PATH="../doc/result/${TIME}_m2"
fi
mkdir -p "${RESULT_PREFIX_PATH}"

run_once() {
  local label=$1
  local strategy=$2
  local result_path="${RESULT_PREFIX_PATH}/celeba_MSAG_${label}_res.txt"
  local log_path="${RESULT_PREFIX_PATH}/celeba_MSAG_${label}.log"
  local w1_fmt
  local w2_fmt
  w1_fmt=$(printf "%.4f" "${SEARCH_W1}")
  w2_fmt=$(printf "%.4f" "${SEARCH_W2}")
  local per_query_path="${result_path%.txt}_w1_${w1_fmt}_w2_${w2_fmt}.txt"
  if [ -s "${result_path}" ] || [ -s "${per_query_path}" ]; then
    echo "[SCRIPT] Skip ${label}: results exist under ${RESULT_PREFIX_PATH}"
    return
  fi
  echo "[SCRIPT] Run ${label}: entry_strategy=${strategy}"
  { time "${EXE_PATH}/S-MSAG" \
      "${MODAL1_BASE_PATH}" \
      "${MODAL2_BASE_PATH}" \
      "${MODAL1_QUERY_PATH}" \
      "${MODAL2_QUERY_PATH}" \
      "${GROUNDTRUTH_PATH}" \
      "${INDEX_PATH}" \
      "${result_path}" \
      "${THREAD_NUM}" \
      "${SEARCH_W1}" \
      "${SEARCH_W2}" \
      "${TOPK_VALUE}" \
      "${GTK_VALUE}" \
      "${L_SEARCH_VALUE}" \
      "${IS_NORM_MODAL1}" \
      "${IS_NORM_MODAL2}" \
      "${IS_SKIP_NUM}" \
      "${SKIP_NUM}" \
      "${IS_MULTI_RESULT_EQUAL}" \
      "${IS_DELETE_ID}" \
      "${DELETE_ID_PATH}" \
      --entry_strategy "${strategy}" \
    --entry_topk "${ENTRY_TOPK}" \
      --centroids_visual "${CENTROIDS_VISUAL_PATH}" \
      --centroids_attr "${CENTROIDS_ATTR_PATH}"; } &> "${log_path}"
  echo "[SCRIPT] Log saved to ${log_path}"
}

run_once baseline 0
run_once target_only 1
run_once adaptive 2
