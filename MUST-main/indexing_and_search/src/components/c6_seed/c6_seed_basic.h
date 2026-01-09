/***************************
@Author: xxx
@Contact: xxx@xxx.com
@File: c6_seed_basic.h
@Time: 2022/4/21 10:10 AM
@Desc:
***************************/

#ifndef GRAPHANNS_C6_SEED_BASIC_H
#define GRAPHANNS_C6_SEED_BASIC_H

#include "../components_basic.h"
#include "../../utils/utils.h"
#include "../../elements/elements.h"
#include <fstream>
#include <string>

class C6SeedBasic : public ComponentsBasic {
protected:
    CStatus init() override {
        auto *s_param = CGRAPH_GET_GPARAM(NPGSearchParam, GA_ALG_NPG_SEARCH_PARAM_KEY)
        model_ = CGRAPH_GET_GPARAM(AnnsModelParam, GA_ALG_MODEL_PARAM_KEY);
        if (nullptr == model_ || nullptr == s_param) {
            CGRAPH_RETURN_ERROR_STATUS("C6SeedBasic get param failed")
        }

        CStatus status = model_->search_meta_modal1_.load(Params.GA_ALG_QUERY_MODAL1_PATH_,
                                                          Params.is_norm_modal1_);
        status += model_->search_meta_modal2_.load(Params.GA_ALG_QUERY_MODAL2_PATH_,
                                                   Params.is_norm_modal2_);
        assert(model_->search_meta_modal1_.num == model_->search_meta_modal2_.num);
        status += model_->train_meta_modal1_.load(Params.GA_ALG_BASE_MODAL1_PATH_, Params.is_norm_modal1_);
        status += model_->train_meta_modal2_.load(Params.GA_ALG_BASE_MODAL2_PATH_, Params.is_norm_modal2_);
        if (Params.is_delete_id_) {
            status += model_->delete_meta_.load(Params.GA_ALG_DELETE_ID_PATH_, 0);
            delete_num_each_query_ = model_->delete_meta_.dim;
        }
        if (Params.entry_strategy_ != 0) {
            if (Params.GA_ALG_CENTROIDS_VISUAL_PATH_.empty()) {
                CGRAPH_RETURN_ERROR_STATUS("centroids_visual path is required for entry_strategy")
            }
            status += model_->centroids_meta_modal1_.load(Params.GA_ALG_CENTROIDS_VISUAL_PATH_, Params.is_norm_modal1_);
            if (Params.entry_strategy_ == 2) {
                if (Params.GA_ALG_CENTROIDS_ATTR_PATH_.empty()) {
                    CGRAPH_RETURN_ERROR_STATUS("centroids_attr path is required for entry_strategy=2")
                }
                status += model_->centroids_meta_modal2_.load(Params.GA_ALG_CENTROIDS_ATTR_PATH_, Params.is_norm_modal2_);
                if (model_->centroids_meta_modal1_.num != model_->centroids_meta_modal2_.num) {
                    CGRAPH_RETURN_ERROR_STATUS("centroid counts mismatch between modal1 and modal2")
                }
            }
            if (model_->centroids_meta_modal1_.dim != model_->train_meta_modal1_.dim) {
                CGRAPH_RETURN_ERROR_STATUS("centroids_visual dim mismatch with modal1 base")
            }
            if (Params.entry_strategy_ == 2 &&
                model_->centroids_meta_modal2_.dim != model_->train_meta_modal2_.dim) {
                CGRAPH_RETURN_ERROR_STATUS("centroids_attr dim mismatch with modal2 base")
            }

            std::string centroid_ids_path = Params.GA_ALG_CENTROID_IDS_PATH_;
            if (centroid_ids_path.empty()) {
                const std::string& cpath = Params.GA_ALG_CENTROIDS_VISUAL_PATH_;
                size_t pos = cpath.find_last_of("/\\");
                std::string dir = (pos == std::string::npos) ? std::string() : cpath.substr(0, pos + 1);
                centroid_ids_path = dir + "centroid_ids.ivecs";
            }
            if (!centroid_ids_path.empty()) {
                std::ifstream in(centroid_ids_path);
                if (in.good()) {
                    status += model_->centroid_ids_meta_.load(centroid_ids_path, 0);
                }
            }
            if (Params.entry_strategy_ == 2 && Params.is_norm_modal2_ == 0 && Params.w2_ != 0) {
                printf("[WARN] entry_strategy=2 with modal2 normalization disabled.\n");
            }
        }
        assert(model_->train_meta_modal1_.num == model_->train_meta_modal2_.num);
        assert(model_->search_meta_modal1_.dim == model_->train_meta_modal1_.dim);
        assert(model_->search_meta_modal2_.dim == model_->train_meta_modal2_.dim);
        if (!status.isOK()) {
            CGRAPH_RETURN_ERROR_STATUS("C6SeedBasic load param failed")
        }

        printf("[PATH] modal 1 query vector path: %s\n", model_->search_meta_modal1_.file_path.c_str());
        printf("[PATH] modal 2 query vector path: %s\n", model_->search_meta_modal2_.file_path.c_str());
        printf("[PARAM] query vector num: %ld\n", model_->search_meta_modal1_.num);
        printf("[PARAM] modal 1 query vector dim: %ld\n", model_->search_meta_modal1_.dim);
        printf("[PARAM] modal 2 query vector dim: %ld\n", model_->search_meta_modal2_.dim);
        if (Params.entry_strategy_ != 0) {
            printf("[PATH] centroids modal1 path: %s\n", model_->centroids_meta_modal1_.file_path.c_str());
            if (Params.entry_strategy_ == 2) {
                printf("[PATH] centroids modal2 path: %s\n", model_->centroids_meta_modal2_.file_path.c_str());
            }
            if (!model_->centroid_ids_meta_.file_path.empty()) {
                printf("[PATH] centroid ids path: %s\n", model_->centroid_ids_meta_.file_path.c_str());
            }
        }
        return CStatus();
    }

protected:
    unsigned search_L_; // candidate pool size for search
    unsigned delete_num_each_query_ = 0;
};

#endif //GRAPHANNS_C6_SEED_BASIC_H
