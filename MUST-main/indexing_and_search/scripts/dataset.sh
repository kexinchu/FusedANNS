#!/bin/sh

# 'DATASET' possible options: "siftsmall", "sift1m", "uqv", "uqv1m", "celeba", "shopping100k" ...
# 'DATASET_SUFFIX' possible options: "tirg_encode", "resnet_encode", "tirg_lstm", "clip4cir_encode", "clip4cir_transformer"
# 'TYPE' possible options: "test", "train"
# 'SUB_DATASET' is only for shopping100k and fashioniq,
# for shopping100k, possible options: "bottoms",
# for fashioniq, possible options: "dress",
DATASET_ROOT=${DATASET_ROOT:-"../doc/dataset"}
DATASET=${DATASET:-"sift1m"}
DATASET_SUFFIX=${DATASET_SUFFIX:-"resnet50_encode"}
TYPE=${TYPE:-"test"}
SUB_DATASET=${SUB_DATASET:-"t-shirt"}
UQV_ROOT=${UQV_ROOT:-"${DATASET_ROOT}/uqv"}
UQV_MODAL2_BASE=${UQV_MODAL2_BASE:-"${UQV_ROOT}/uqv_modal2_base.ivecs"}
UQV_MODAL2_QUERY=${UQV_MODAL2_QUERY:-"${UQV_ROOT}/uqv_modal2_query.ivecs"}

dataset() {
  case $DATASET in
    siftsmall)
      MODAL1_BASE_PATH="${DATASET_ROOT}/${DATASET}/siftsmall_base.fvecs"
      MODAL2_BASE_PATH="${DATASET_ROOT}/${DATASET}/int_attribute_siftsmall_base.ivecs"
      MODAL1_QUERY_PATH="${DATASET_ROOT}/${DATASET}/siftsmall_query.fvecs"
      MODAL2_QUERY_PATH="${DATASET_ROOT}/${DATASET}/int_attribute_siftsmall_query.ivecs"
      GROUNDTRUTH_PATH="${DATASET_ROOT}/${DATASET}/siftsmall_groundtruth.ivecs"
      DELETE_ID_PATH="NULL"
      ;;
    deep1m|deep2m|deep4m|deep8m|deep16m|deep32m|deep64m|deep128m)
      MODAL1_BASE_PATH="${DATASET_ROOT}/deep/${DATASET}/${DATASET}_modal1_base.fvecs"
      MODAL2_BASE_PATH="${DATASET_ROOT}/deep/${DATASET}/${DATASET}_modal2_base.ivecs"
      MODAL1_QUERY_PATH="${DATASET_ROOT}/deep/deep_modal1_query.fvecs"
      MODAL2_QUERY_PATH="${DATASET_ROOT}/deep/deep_modal2_query.ivecs"
      GROUNDTRUTH_PATH="${DATASET_ROOT}/deep/${DATASET}/${DATASET}_gt.ivecs"
      DELETE_ID_PATH="NULL"
      ;;
    sift1m|uqv1m|msong1m)
      MODAL1_BASE_PATH="${DATASET_ROOT}/${DATASET}/${DATASET}_modal1_base.fvecs"
      MODAL2_BASE_PATH="${DATASET_ROOT}/${DATASET}/${DATASET}_modal2_base.ivecs"
      MODAL1_QUERY_PATH="${DATASET_ROOT}/${DATASET}/${DATASET}_modal1_query.fvecs"
      MODAL2_QUERY_PATH="${DATASET_ROOT}/${DATASET}/${DATASET}_modal2_query.ivecs"
      GROUNDTRUTH_PATH="${DATASET_ROOT}/${DATASET}/${DATASET}_gt.ivecs"
#      GROUNDTRUTH_PATH="/home/xxxxx/ANNS/dataset/sift/sift_groundtruth.ivecs"
      DELETE_ID_PATH="NULL"
    ;;
    uqv)
      MODAL1_BASE_PATH="${UQV_ROOT}/uqv_base.fvecs"
      MODAL1_QUERY_PATH="${UQV_ROOT}/uqv_query.fvecs"
      GROUNDTRUTH_PATH="${UQV_ROOT}/uqv_groundtruth.ivecs"
      MODAL2_BASE_PATH="${UQV_MODAL2_BASE}"
      MODAL2_QUERY_PATH="${UQV_MODAL2_QUERY}"
      DELETE_ID_PATH="NULL"
      if [ ! -f "${MODAL2_BASE_PATH}" ] || [ ! -f "${MODAL2_QUERY_PATH}" ]; then
        echo "[DATASET] uqv modal2 files not found."
        echo "[DATASET] Set UQV_MODAL2_BASE/UQV_MODAL2_QUERY or place uqv_modal2_base.ivecs/uqv_modal2_query.ivecs under ${UQV_ROOT}."
        exit 1
      fi
    ;;
    celeba)
      case $TYPE in
        test)
          MODAL1_BASE_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET_SUFFIX}/${DATASET}_modal1_base.fvecs"
          MODAL2_BASE_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET}_modal2_base.ivecs"
          MODAL2_QUERY_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET}_modal2_query.ivecs"
          GROUNDTRUTH_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET}_gt.ivecs"
          DELETE_ID_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET}_delete_id.ivecs"
          case $DATASET_SUFFIX in
            tirg_encode|clip4cir_encode)
              MODAL1_QUERY_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET_SUFFIX}/${DATASET}_modal1_2_query.fvecs"
#              MODAL1_QUERY_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET_SUFFIX}/${DATASET}_modal1_base.fvecs"
            ;;
            resnet_encode|resnet50_encode|raw32)
              MODAL1_QUERY_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET_SUFFIX}/${DATASET}_modal1_query.fvecs"
          esac
        ;;
        train)
          MODAL1_BASE_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET_SUFFIX}/${DATASET}_modal1_base.fvecs"
          MODAL2_BASE_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET_SUFFIX}/${DATASET}_modal2_base.ivecs"
          MODAL2_QUERY_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET_SUFFIX}/${DATASET}_modal2_query.ivecs"
          GROUNDTRUTH_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET_SUFFIX}/${DATASET}_gt.ivecs"
          DELETE_ID_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET}_delete_id.ivecs(padding)"
          case $DATASET_SUFFIX in
            tirg_encode)
              MODAL1_QUERY_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET_SUFFIX}/${DATASET}_modal1_2_query.fvecs"
            ;;
            resnet_encode|raw32)
              MODAL1_QUERY_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET_SUFFIX}/${DATASET}_modal1_query.fvecs"
          esac
      esac
    ;;
    celeba+)
      case $TYPE in
        test)
          MODAL1_BASE_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/celeba_modal1_base.fvecs"
          MODAL2_BASE_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/celeba_modal2_base.ivecs"
          MODAL3_BASE_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/celeba_modal3_base.fvecs"
          MODAL4_BASE_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/celeba_modal4_base.fvecs"
          MODAL1_QUERY_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/celeba_modal1_query.fvecs"
          MODAL2_QUERY_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/celeba_modal2_query.ivecs"
          MODAL3_QUERY_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/celeba_modal3_query.fvecs"
          MODAL4_QUERY_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/celeba_modal4_query.fvecs"
          GROUNDTRUTH_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/celeba_gt.ivecs"
          DELETE_ID_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/celeba_delete_id.ivecs"
        ;;
        train)
          MODAL1_BASE_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET_SUFFIX}/${DATASET}_modal1_base.fvecs"
          MODAL2_BASE_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET_SUFFIX}/${DATASET}_modal2_base.ivecs"
          MODAL2_QUERY_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET_SUFFIX}/${DATASET}_modal2_query.ivecs"
          GROUNDTRUTH_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET_SUFFIX}/${DATASET}_gt.ivecs"
          DELETE_ID_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET}_delete_id.ivecs(padding)"
          case $DATASET_SUFFIX in
            tirg_encode)
              MODAL1_QUERY_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET_SUFFIX}/${DATASET}_modal1_2_query.fvecs"
            ;;
            resnet_encode)
              MODAL1_QUERY_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET_SUFFIX}/${DATASET}_modal1_query.fvecs"
          esac
      esac
    ;;
    shopping100k)
      case $TYPE in
        test)
          MODAL1_BASE_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${SUB_DATASET}/${DATASET_SUFFIX}/${DATASET}_${SUB_DATASET}_modal1_base.fvecs"
          MODAL2_BASE_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${SUB_DATASET}/${DATASET}_${SUB_DATASET}_modal2_base.ivecs"
          MODAL2_QUERY_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${SUB_DATASET}/${DATASET}_${SUB_DATASET}_modal2_query.ivecs"
          GROUNDTRUTH_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${SUB_DATASET}/${DATASET}_${SUB_DATASET}_gt.ivecs"
          DELETE_ID_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${SUB_DATASET}/${DATASET}_${SUB_DATASET}_delete_id.ivecs"
          case $DATASET_SUFFIX in
            tirg_encode)
              MODAL1_QUERY_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${SUB_DATASET}/${DATASET_SUFFIX}/${DATASET}_${SUB_DATASET}_modal1_2_query.fvecs"
            ;;
            resnet_encode)
              MODAL1_QUERY_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${SUB_DATASET}/${DATASET_SUFFIX}/${DATASET}_${SUB_DATASET}_modal1_query.fvecs"
          esac
        ;;
        train)
          MODAL1_BASE_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET_SUFFIX}/${DATASET}_modal1_base.fvecs"
          MODAL2_BASE_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET_SUFFIX}/${DATASET}_modal2_base.ivecs"
          MODAL2_QUERY_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET_SUFFIX}/${DATASET}_modal2_query.ivecs"
          GROUNDTRUTH_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET_SUFFIX}/${DATASET}_gt.ivecs"
          DELETE_ID_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET}_delete_id.ivecs(padding)"
          case $DATASET_SUFFIX in
            tirg_encode)
              MODAL1_QUERY_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET_SUFFIX}/${DATASET}_modal1_2_query.fvecs"
            ;;
            resnet_encode)
              MODAL1_QUERY_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET_SUFFIX}/${DATASET}_modal1_query.fvecs"
          esac
      esac
    ;;
    mitstates)
      case $TYPE in
        test)
          MODAL1_BASE_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET_SUFFIX}/${DATASET}_modal1_base.fvecs"
          MODAL2_BASE_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET_SUFFIX}/${DATASET}_modal2_base.fvecs"
          MODAL2_QUERY_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET_SUFFIX}/${DATASET}_modal2_query.fvecs"
          GROUNDTRUTH_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET}_gt.ivecs"
          DELETE_ID_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET}_delete_id.ivecs"
          case $DATASET_SUFFIX in
            tirg_lstm|clip4cir_transformer|clip4cir_lstm|tirg_transformer)
              MODAL1_QUERY_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET_SUFFIX}/${DATASET}_modal1_2_query.fvecs"
            ;;
            resnet_lstm|resnet50_transformer|resnet50_lstm|resnet_transformer)
              MODAL1_QUERY_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET_SUFFIX}/${DATASET}_modal1_query.fvecs"
          esac
        ;;
        train)
          MODAL1_BASE_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET_SUFFIX}/${DATASET}_modal1_base.fvecs"
          MODAL2_BASE_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET_SUFFIX}/${DATASET}_modal2_base.fvecs"
          MODAL2_QUERY_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET_SUFFIX}/${DATASET}_modal2_query.fvecs"
          GROUNDTRUTH_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET_SUFFIX}/${DATASET}_gt.ivecs"
          DELETE_ID_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET}_delete_id.ivecs(padding)"
          case $DATASET_SUFFIX in
            tirg_lstm)
              MODAL1_QUERY_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET_SUFFIX}/${DATASET}_modal1_2_query.fvecs"
            ;;
            resnet_lstm)
              MODAL1_QUERY_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET_SUFFIX}/${DATASET}_modal1_query.fvecs"
          esac
      esac
    ;;
    fashioniq)
      case $TYPE in
        test)
          MODAL1_BASE_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${SUB_DATASET}/${DATASET_SUFFIX}/${DATASET}_${SUB_DATASET}_modal1_base.fvecs"
          MODAL2_BASE_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${SUB_DATASET}/${DATASET}_${SUB_DATASET}_modal2_base.fvecs"
          MODAL2_QUERY_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${SUB_DATASET}/${DATASET}_${SUB_DATASET}_modal2_query.fvecs"
          GROUNDTRUTH_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${SUB_DATASET}/${DATASET}_${SUB_DATASET}_gt.ivecs"
          DELETE_ID_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${SUB_DATASET}/${DATASET}_${SUB_DATASET}_delete_id.ivecs"
          case $DATASET_SUFFIX in
            tirg_lstm)
              MODAL1_QUERY_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${SUB_DATASET}/${DATASET_SUFFIX}/${DATASET}_${SUB_DATASET}_modal1_2_query.fvecs"
            ;;
            resnet_lstm)
              MODAL1_QUERY_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${SUB_DATASET}/${DATASET_SUFFIX}/${DATASET}_${SUB_DATASET}_modal1_query.fvecs"
          esac
        ;;
        train)
          MODAL1_BASE_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET_SUFFIX}/${DATASET}_modal1_base.fvecs"
          MODAL2_BASE_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET_SUFFIX}/${DATASET}_modal2_base.fvecs"
          MODAL2_QUERY_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET_SUFFIX}/${DATASET}_modal2_query.fvecs"
          GROUNDTRUTH_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET_SUFFIX}/${DATASET}_gt.ivecs"
          DELETE_ID_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET}_delete_id.ivecs(padding)"
          case $DATASET_SUFFIX in
            tirg_lstm)
              MODAL1_QUERY_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET_SUFFIX}/${DATASET}_modal1_2_query.fvecs"
            ;;
            resnet_lstm)
              MODAL1_QUERY_PATH="${DATASET_ROOT}/${DATASET}/${TYPE}/${DATASET_SUFFIX}/${DATASET}_modal1_query.fvecs"
          esac
      esac
    ;;
    uqv_syn)
      S_TAG=${SUB_DATASET#s_}
      MODAL1_BASE_PATH="${DATASET_ROOT}/${DATASET}/${SUB_DATASET}/random_base_n100000_d32_c10_s${S_TAG}.fvecs"
      MODAL2_BASE_PATH="${DATASET_ROOT}/${DATASET}/${SUB_DATASET}/random_base_n100000_d32_c10_s${S_TAG}.fvecs"
      MODAL1_QUERY_PATH="${DATASET_ROOT}/${DATASET}/${SUB_DATASET}/random_query_n1000_d32_c10_s${S_TAG}.fvecs"
      MODAL2_QUERY_PATH="${DATASET_ROOT}/${DATASET}/${SUB_DATASET}/random_query_n1000_d32_c10_s${S_TAG}.fvecs"
      GROUNDTRUTH_PATH="${DATASET_ROOT}/${DATASET}/${SUB_DATASET}/random_ground_truth_n1000_d32_c10_s${S_TAG}.ivecs"
      DELETE_ID_PATH="NULL"
    ;;
    ImageText1M|AudioText1M|VideoText1M)
      MODAL1_BASE_PATH="/home/xxxxx/ANNS/xxx/GraphANNS/doc/dataset/sift1m/sift1m_modal1_base.fvecs"
      MODAL2_BASE_PATH="/home/xxxxx/ANNS/xxx/int_label_base.ivecs"
      MODAL1_QUERY_PATH="/home/xxxxx/ANNS/xxx/GraphANNS/doc/dataset/sift1m/sift1m_modal1_query.fvecs"
      MODAL2_QUERY_PATH="/home/xxxxx/ANNS/xxx/int_label_query.ivecs"
      GROUNDTRUTH_PATH="/home/xxxxx/ANNS/dataset/sift/label_sift_groundtruth.ivecs"
      DELETE_ID_PATH="/home/xxxxx/ANNS/xxx/GraphANNS/doc/dataset/shopping100k/test/bottoms/shopping100k_bottoms_delete_id.ivecs"
  esac
}
