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
source "${SCRIPT_DIR}/param.sh"

param_${DATASET}_${STRATEGY}

TIME=(`date +%Y_%m_%d`)
RESULT_PREFIX_PATH="../doc/result/${TIME}"
SEARCH_RESULT_PATH=${RESULT_PREFIX_PATH}/${DATASET}_${STRATEGY}_res.txt
#SEARCH_RESULT_PATH=" "
EXTRA_CMAKE_ARGS=${EXTRA_CMAKE_ARGS:-}
MSAG_BUILD_EXTRA_ARGS=${MSAG_BUILD_EXTRA_ARGS:-}
MSAG_SEARCH_EXTRA_ARGS=${MSAG_SEARCH_EXTRA_ARGS:-}

# Allow search-time overrides without rebuilding the index.
SEARCH_MODAL1_BASE_PATH=${SEARCH_MODAL1_BASE_PATH:-$MODAL1_BASE_PATH}
SEARCH_MODAL2_BASE_PATH=${SEARCH_MODAL2_BASE_PATH:-$MODAL2_BASE_PATH}
SEARCH_MODAL1_QUERY_PATH=${SEARCH_MODAL1_QUERY_PATH:-$MODAL1_QUERY_PATH}
SEARCH_MODAL2_QUERY_PATH=${SEARCH_MODAL2_QUERY_PATH:-$MODAL2_QUERY_PATH}
SEARCH_GROUNDTRUTH_PATH=${SEARCH_GROUNDTRUTH_PATH:-$GROUNDTRUTH_PATH}
SEARCH_INDEX_PATH=${SEARCH_INDEX_PATH:-$INDEX_PATH}
SEARCH_INDEX_PATH_MODAL1=${SEARCH_INDEX_PATH_MODAL1:-$INDEX_PATH_MODAL1}
SEARCH_INDEX_PATH_MODAL2=${SEARCH_INDEX_PATH_MODAL2:-$INDEX_PATH_MODAL2}
SEARCH_W1=${SEARCH_W1:-$W1}
SEARCH_W2=${SEARCH_W2:-$W2}
SEARCH_TOPK=${SEARCH_TOPK:-$TOPK}
SEARCH_GTK=${SEARCH_GTK:-$GTK}

SEARCH_L_VALUES=("${L_search[@]}")
if [[ -n "${SEARCH_L:-}" ]]; then
  SEARCH_L_VALUES=("${SEARCH_L}")
fi

SEARCH_CANDIDATE_TOPK_VALUES=("${CANDIDATE_TOPK[@]}")
if [[ -n "${SEARCH_CANDIDATE_TOPK:-}" ]]; then
  SEARCH_CANDIDATE_TOPK_VALUES=("${SEARCH_CANDIDATE_TOPK}")
fi

SEARCH_TOPK_VALUES=("${TOPK[@]}")
if [[ -n "${SEARCH_TOPK_LIST:-}" ]]; then
  SEARCH_TOPK_VALUES=(${SEARCH_TOPK_LIST})
fi

if [ ! -e ${RESULT_PREFIX_PATH} ]
then
mkdir -p ${RESULT_PREFIX_PATH}
echo "${RESULT_PREFIX_PATH} is set."
fi

if [ ! -e ${INDEX_PREFIX_PATH} ]
then
mkdir -p ${INDEX_PREFIX_PATH}
echo "${INDEX_PREFIX_PATH} is set."
fi

print_usage_and_exit() {
  echo "[SCRIPT] Usage: ./run.sh [debug/release] [build_${STRATEGY}/search_$STRATEGY}]"
  exit 1
}

check_dir_and_make_if_absent() {
  local dir=$1
  if [ -d $dir ]; then
    echo "[SCRIPT] Directory $dir is already exit. Remove or rename it and then re-run."
    exit 1
  else
    mkdir -p ${dir}
  fi
}

# compiling
case $1 in
  debug)
    cmake -DCMAKE_BUILD_TYPE=Debug .. -B ../debug ${EXTRA_CMAKE_ARGS}
    EXE_PATH=../debug
  ;;
  release)
    cmake -DCMAKE_BUILD_TYPE=Release .. -B ../release ${EXTRA_CMAKE_ARGS}
    EXE_PATH=../release
  ;;
  *)
    print_usage_and_exit
  ;;
esac
pushd $EXE_PATH
make -j
popd

case $2 in
  build_MSAG)
    echo "[SCRIPT] Building MultimodalSimilarityAggregationGraph index on ${DATASET} ..."
    time $EXE_PATH/T-MSAG \
      ${SEARCH_MODAL1_BASE_PATH} \
      ${SEARCH_MODAL2_BASE_PATH} \
      ${INDEX_PATH} \
      ${BUILD_THREAD_NUM} \
      ${W1} \
      ${W2} \
      ${L_candidate} \
      ${R_neighbor} \
      ${C_neighbor} \
      ${k_init_graph} \
      ${nn_size} \
      ${rnn_size} \
      ${pool_size} \
      ${iter} \
      ${sample_num} \
      ${graph_quality_threshold} \
      ${IS_NORM_MODAL1} \
      ${IS_NORM_MODAL2} \
      ${IS_SKIP_NUM} \
      ${SKIP_NUM} \
      ${MSAG_BUILD_EXTRA_ARGS} >> ${RESULT_PREFIX_PATH}/build_${STRATEGY}.log
  ;;
  build_MABG)
    echo "[SCRIPT] Modal1: Building MultichannelAggregationByGraph index on ${DATASET} ..."
    time $EXE_PATH/T-MSAG \
      ${SEARCH_MODAL1_BASE_PATH} \
      ${SEARCH_MODAL2_BASE_PATH} \
      ${INDEX_PATH_MODAL1} \
      ${BUILD_THREAD_NUM} \
      ${W1} \
      0 \
      ${L_candidate} \
      ${R_neighbor} \
      ${C_neighbor} \
      ${k_init_graph} \
      ${nn_size} \
      ${rnn_size} \
      ${pool_size} \
      ${iter} \
      ${sample_num} \
      ${graph_quality_threshold} \
      ${IS_NORM_MODAL1} \
      ${IS_NORM_MODAL2} \
      ${IS_SKIP_NUM} \
      ${SKIP_NUM} >> ${RESULT_PREFIX_PATH}/build_${STRATEGY}.log
    echo "[SCRIPT] Modal2: Building MultichannelAggregationByGraph index on ${DATASET} ..."
    time $EXE_PATH/T-MSAG \
      ${SEARCH_MODAL1_BASE_PATH} \
      ${SEARCH_MODAL2_BASE_PATH} \
      ${INDEX_PATH_MODAL2} \
      ${BUILD_THREAD_NUM} \
      0 \
      ${W2} \
      ${L_candidate} \
      ${R_neighbor} \
      ${C_neighbor} \
      ${k_init_graph} \
      ${nn_size} \
      ${rnn_size} \
      ${pool_size} \
      ${iter} \
      ${sample_num} \
      ${graph_quality_threshold} \
      ${IS_NORM_MODAL1} \
      ${IS_NORM_MODAL2} \
      ${IS_SKIP_NUM} \
      ${SKIP_NUM} >> ${RESULT_PREFIX_PATH}/build_${STRATEGY}.log
  ;;
  search_MABG)
    echo "[SCRIPT] Searching by MultichannelAggregationByGraph on ${DATASET} ..."
    for EACH_CANDIDATE_TOPK in "${SEARCH_CANDIDATE_TOPK_VALUES[@]}"
    do
    L_search=${EACH_CANDIDATE_TOPK}
    time $EXE_PATH/S-MABG \
      ${SEARCH_MODAL1_BASE_PATH} \
      ${SEARCH_MODAL2_BASE_PATH} \
      ${SEARCH_MODAL1_QUERY_PATH} \
      ${SEARCH_MODAL2_QUERY_PATH} \
      ${SEARCH_GROUNDTRUTH_PATH} \
      ${SEARCH_INDEX_PATH_MODAL1} \
      ${SEARCH_INDEX_PATH_MODAL2} \
      ${SEARCH_RESULT_PATH} \
      ${SEARCH_THREAD_NUM} \
      ${SEARCH_W1} \
      ${SEARCH_W2} \
      ${SEARCH_TOPK} \
      ${SEARCH_GTK} \
      ${L_search} \
      ${IS_NORM_MODAL1} \
      ${IS_NORM_MODAL2} \
      ${IS_SKIP_NUM} \
      ${SKIP_NUM} \
      ${IS_MULTI_RESULT_EQUAL} \
      ${IS_DELETE_ID} \
      ${DELETE_ID_PATH} \
      ${EACH_CANDIDATE_TOPK} >> ${RESULT_PREFIX_PATH}/search_${STRATEGY}.log
      done
  ;;
  search_MSAG)
    echo "[SCRIPT] Searching by MultimodalSimilarityAggregationGraph on ${DATASET} ..."
    for EACH_L_search in "${SEARCH_L_VALUES[@]}"
    do
    time $EXE_PATH/S-MSAG \
      ${MODAL1_BASE_PATH} \
      ${MODAL2_BASE_PATH} \
      ${SEARCH_MODAL1_QUERY_PATH} \
      ${SEARCH_MODAL2_QUERY_PATH} \
      ${SEARCH_GROUNDTRUTH_PATH} \
      ${SEARCH_INDEX_PATH} \
      ${SEARCH_RESULT_PATH} \
      ${SEARCH_THREAD_NUM} \
      ${SEARCH_W1} \
      ${SEARCH_W2} \
      ${SEARCH_TOPK} \
      ${SEARCH_GTK} \
      ${EACH_L_search} \
      ${IS_NORM_MODAL1} \
      ${IS_NORM_MODAL2} \
      ${IS_SKIP_NUM} \
      ${SKIP_NUM} \
      ${IS_MULTI_RESULT_EQUAL} \
      ${IS_DELETE_ID} \
      ${DELETE_ID_PATH} \
      ${MSAG_SEARCH_EXTRA_ARGS} >> ${RESULT_PREFIX_PATH}/search_${STRATEGY}.log
      done
  ;;
  search_MSAB)
    echo "[SCRIPT] Searching by MultimodalSimilarityAggregationBruteforce on ${DATASET} ..."
    for EACH_TOPK in "${SEARCH_TOPK_VALUES[@]}"
    do
    time $EXE_PATH/S-MSAB \
      ${MODAL1_BASE_PATH} \
      ${MODAL2_BASE_PATH} \
      ${SEARCH_MODAL1_QUERY_PATH} \
      ${SEARCH_MODAL2_QUERY_PATH} \
      ${SEARCH_GROUNDTRUTH_PATH} \
      ${SEARCH_RESULT_PATH} \
      ${SEARCH_THREAD_NUM} \
      ${SEARCH_W1} \
      ${SEARCH_W2} \
      ${EACH_TOPK} \
      ${SEARCH_GTK} \
      ${IS_NORM_MODAL1} \
      ${IS_NORM_MODAL2} \
      ${IS_SKIP_NUM} \
      ${SKIP_NUM} \
      ${IS_MULTI_RESULT_EQUAL} \
      ${IS_DELETE_ID} \
      ${DELETE_ID_PATH} >> ${RESULT_PREFIX_PATH}/search_${STRATEGY}.log
      done
  ;;
  search_MABB)
    echo "[SCRIPT] Searching by MultichannelAggregationByBruteforce on ${DATASET} ..."
    for EACH_CANDIDATE_TOPK in "${SEARCH_CANDIDATE_TOPK_VALUES[@]}"
    do
    time $EXE_PATH/S-MABB \
      ${MODAL1_BASE_PATH} \
      ${MODAL2_BASE_PATH} \
      ${SEARCH_MODAL1_QUERY_PATH} \
      ${SEARCH_MODAL2_QUERY_PATH} \
      ${SEARCH_GROUNDTRUTH_PATH} \
      ${SEARCH_RESULT_PATH} \
      ${SEARCH_THREAD_NUM} \
      ${SEARCH_W1} \
      ${SEARCH_W2} \
      ${SEARCH_TOPK} \
      ${SEARCH_GTK} \
      ${IS_NORM_MODAL1} \
      ${IS_NORM_MODAL2} \
      ${IS_SKIP_NUM} \
      ${SKIP_NUM} \
      ${IS_MULTI_RESULT_EQUAL} \
      ${IS_DELETE_ID} \
      ${DELETE_ID_PATH} \
      ${EACH_CANDIDATE_TOPK} >> ${RESULT_PREFIX_PATH}/search_${STRATEGY}.log
    done
  ;;
esac
