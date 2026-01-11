/***************************
@Author: xxx
@Contact: xxx@xxx.com
@File: c7_routing_kgraph.h
@Time: 2022/4/8 8:37 AM
@Desc: greedy route (like 'KGraph' algorithm)
***************************/

#ifndef GRAPHANNS_C7_ROUTING_KGRAPH_H
#define GRAPHANNS_C7_ROUTING_KGRAPH_H

#include "../c7_routing_basic.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <utility>

class C7RoutingKGraph : public C7RoutingBasic {
public:
    DAnnFuncType prepareParam() override {
        auto *s_param = CGRAPH_GET_GPARAM(NPGSearchParam, GA_ALG_NPG_SEARCH_PARAM_KEY);
        auto *a_param = CGRAPH_GET_GPARAM(AlgParamBasic, GA_ALG_PARAM_BASIC_KEY);
        model_ = CGRAPH_GET_GPARAM(AnnsModelParam, GA_ALG_MODEL_PARAM_KEY);
        if (nullptr == model_ || nullptr == s_param) {
            return DAnnFuncType::ANN_PREPARE_ERROR;
        }

        num_ = model_->train_meta_modal1_.num;
        dim1_ = model_->train_meta_modal1_.dim;
        dim2_ = model_->train_meta_modal2_.dim;
        data_modal1_ = model_->train_meta_modal1_.data;
        data_modal2_ = model_->train_meta_modal2_.data;
        search_L_ = s_param->search_L;
        K_ = a_param->top_k;
        query_id_ = s_param->query_id;
        query_modal1_ = model_->search_meta_modal1_.data;
        query_modal2_ = model_->search_meta_modal2_.data;
        if (Params.is_delete_id_) {
            delete_num_each_query_ = model_->delete_meta_.dim;
        }
        return DAnnFuncType::ANN_SEARCH;
    }

    CStatus search() override {
        auto s_param = CGRAPH_GET_GPARAM(NPGSearchParam, GA_ALG_NPG_SEARCH_PARAM_KEY);
        if (nullptr == s_param) {
            CGRAPH_RETURN_ERROR_STATUS("C7RoutingKGraph search get param failed")
        }

        static int debug_print_count = 0;
        std::vector<char> flags(num_, 0);
        res_.clear();

        IDType best_ep = 0;
        if (!s_param->sp.empty()) {
            best_ep = s_param->sp[0].id_;
        }
        DistResType best_score = 0;
        int topk_inserted = 0;
        int topk_requested = 0;
        bool used_entry_topk = false;
        if (Params.entry_strategy_ == 1 || Params.entry_strategy_ == 2) {
            const auto &centroids1 = model_->centroids_meta_modal1_;
            const auto &centroids2 = model_->centroids_meta_modal2_;
            if (centroids1.data != nullptr && centroids1.num > 0 &&
                (Params.entry_strategy_ == 1 ||
                 (centroids2.data != nullptr && centroids2.num > 0))) {
                const VecValType1 *q1 = query_modal1_ + ((size_t)query_id_ * (size_t)dim1_);
                const VecValType2 *q2 = query_modal2_ + ((size_t)query_id_ * (size_t)dim2_);
                std::vector<std::pair<DistResType, IDType>> ep_candidates;
                ep_candidates.reserve(centroids1.num);
                IDType best_visual_centroid = 0;
                DistResType best_visual_score = std::numeric_limits<DistResType>::max();
                DistResType1 best_visual_raw = 0;
                for (IDType i = 0; i < centroids1.num; ++i) {
                    DistResType1 d1 = 0;
                    DistResType2 d2 = 0;
                    DistResType dist = 0;
                    bool has_d1 = false;
                    const VecValType1 *c1 = centroids1.data + (size_t)i * (size_t)centroids1.dim;
                    if (Params.entry_strategy_ == 1) {
                        dist_op_.dist_op1_.calculate(q1, c1, dim1_, centroids1.dim, d1);
                        has_d1 = true;
                        dist = d1 * dist_op_.weight_1_;
                    } else {
                        if (dist_op_.weight_1_ != 0) {
                            dist_op_.dist_op1_.calculate(q1, c1, dim1_, centroids1.dim, d1);
                            has_d1 = true;
                        }
                        const VecValType2 *c2 = centroids2.data + (size_t)i * (size_t)centroids2.dim;
                        if (dist_op_.weight_2_ != 0) {
                            dist_op_.dist_op2_.calculate(q2, c2, dim2_, centroids2.dim, d2);
                        }
                        dist = d1 * dist_op_.weight_1_ + d2 * dist_op_.weight_2_;
                    }
                    if (!has_d1) {
                        dist_op_.dist_op1_.calculate(q1, c1, dim1_, centroids1.dim, d1);
                    }
                    DistResType visual_score = d1 * dist_op_.weight_1_;
                    if (visual_score < best_visual_score) {
                        best_visual_score = visual_score;
                        best_visual_centroid = i;
                        best_visual_raw = d1;
                    }
                    ep_candidates.emplace_back(dist, i);
                }
                std::sort(ep_candidates.begin(), ep_candidates.end(),
                          [](const auto &a, const auto &b) { return a.first < b.first; });
                topk_requested = static_cast<int>(ep_candidates.size());
                if (Params.entry_topk_ > 0 &&
                    topk_requested > static_cast<int>(Params.entry_topk_)) {
                    topk_requested = static_cast<int>(Params.entry_topk_);
                }
                if (topk_requested > static_cast<int>(search_L_)) {
                    topk_requested = static_cast<int>(search_L_);
                }
                s_param->sp.assign(search_L_ + 1, NeighborFlag(0,
                                                              std::numeric_limits<DistResType>::max(),
                                                              false));
                for (int i = 0; i < topk_requested; ++i) {
                    IDType centroid_idx = ep_candidates[i].second;
                    IDType mapped_id = centroid_idx;
                    if (model_->centroid_ids_meta_.data != nullptr &&
                        model_->centroid_ids_meta_.num > centroid_idx &&
                        model_->centroid_ids_meta_.dim > 0) {
                        mapped_id = model_->centroid_ids_meta_.data[
                            (size_t)centroid_idx * (size_t)model_->centroid_ids_meta_.dim];
                    }
                    if (mapped_id >= num_) {
                        continue;
                    }
                    DistResType ep_dist = 0;
                    dist_op_.calculate(q1,
                                      data_modal1_ + (size_t)mapped_id * (size_t)dim1_,
                                      dim1_, dim1_,
                                      q2,
                                      data_modal2_ + (size_t)mapped_id * (size_t)dim2_,
                                      dim2_, dim2_, ep_dist);
                    NeighborFlag nn(mapped_id, ep_dist, true);
                    int r = InsertIntoPool(s_param->sp.data(), search_L_, nn);
                    if (r <= static_cast<int>(search_L_)) {
                        if (topk_inserted == 0) {
                            best_ep = mapped_id;
                            best_score = ep_candidates[i].first;
                        }
                        ++topk_inserted;
                    }
                }
                used_entry_topk = topk_inserted > 0;
                if (query_id_ == 18254) {
                    std::cout << "--- Debug QID 18254 Top-" << Params.entry_topk_ << " EPs ---" << std::endl;
                    for (size_t i = 0; i < ep_candidates.size() && i < Params.entry_topk_; ++i) {
                        IDType centroid_idx = ep_candidates[i].second;
                        IDType mapped_id = centroid_idx;
                        if (model_->centroid_ids_meta_.data != nullptr &&
                            model_->centroid_ids_meta_.num > centroid_idx &&
                            model_->centroid_ids_meta_.dim > 0) {
                            mapped_id = model_->centroid_ids_meta_.data[
                                (size_t)centroid_idx * (size_t)model_->centroid_ids_meta_.dim];
                        }
                        std::cout << "Rank " << i
                                  << ": Centroid " << centroid_idx
                                  << " (Score: " << ep_candidates[i].first
                                  << ", BaseID: " << mapped_id << ")" << std::endl;
                    }
                    std::cout << "Visual Best Centroid: " << best_visual_centroid
                              << " (Score: " << best_visual_score
                              << ", Raw: " << best_visual_raw << ")" << std::endl;
                }
            }
        }
        std::cout << "[DEBUG] Strategy: " << Params.entry_strategy_
                  << ", Selected Entry Point ID: " << best_ep << std::endl;
        if (used_entry_topk) {
            std::cout << "[DEBUG] Top-1 EP: " << best_ep
                      << " (Score: " << best_score << ")" << std::endl;
            std::cout << "[DEBUG] Inserted Top-" << topk_inserted
                      << " Entry Points (requested " << topk_requested << ")." << std::endl;
        }

        unsigned k = 0;
        while (k < (int) search_L_) {
            unsigned nk = search_L_;

            if (s_param->sp[k].flag_) {
                s_param->sp[k].flag_ = false;
                IDType n = s_param->sp[k].id_;

                for (unsigned int id : model_->graph_m_[n]) {
                    if (flags[id]) continue;
                    flags[id] = 1;
                    bool is_delete = false;
                    if (delete_num_each_query_) {
                        for (IDType k = 0; k < delete_num_each_query_; k++) {
                            if (id == model_->delete_meta_.data[s_param->query_id * delete_num_each_query_ + k]) {
                                is_delete = true;
                                break;
                            }
                        }
                    }
                    if (is_delete) continue;

                    DistResType dist = 0;
                    dist_op_.calculate(query_modal1_ + ((size_t)query_id_ * (size_t)dim1_),
                                      data_modal1_ + (size_t)id * (size_t)dim1_,
                                      dim1_, dim1_,
                                       query_modal2_ + ((size_t)query_id_ * (size_t)dim2_),
                                       data_modal2_ + (size_t)id * (size_t)dim2_,
                                      dim2_, dim2_, dist);
                    if (query_id_ == 0 && debug_print_count < 10) {
                        DistResType1 d1_raw = 0;
                        DistResType2 d2_raw = 0;
                        dist_op_.dist_op1_.calculate(query_modal1_ + ((size_t)query_id_ * (size_t)dim1_),
                                                     data_modal1_ + (size_t)id * (size_t)dim1_,
                                                     dim1_, dim1_, d1_raw);
                        dist_op_.dist_op2_.calculate(query_modal2_ + ((size_t)query_id_ * (size_t)dim2_),
                                                     data_modal2_ + (size_t)id * (size_t)dim2_,
                                                     dim2_, dim2_, d2_raw);
                        DistResType final_dist = d1_raw * dist_op_.weight_1_ +
                                                 (DistResType) d2_raw * dist_op_.weight_2_;
                        std::cout << "[DEBUG] q=0 id=" << id
                                  << " | D1: " << d1_raw
                                  << " | D2: " << d2_raw
                                  << " | Final: " << final_dist << std::endl;
                        ++debug_print_count;
                    }

                    if (dist >= s_param->sp[search_L_ - 1].distance_) continue;
                    NeighborFlag nn(id, dist, true);
                    int r = InsertIntoPool(s_param->sp.data(), search_L_, nn);

                    if (r < nk) nk = r;
                }
            }
            nk <= k ? (k = nk) : (++k);
        }

        res_.reserve(K_);
        for (size_t i = 0; i < K_; i++) {
            res_.push_back(s_param->sp[i].id_);
        }
        return CStatus();
    }

    CStatus refreshParam() override {
        auto a_param = CGRAPH_GET_GPARAM(AlgParamBasic, GA_ALG_PARAM_BASIC_KEY);
        CGRAPH_ASSERT_NOT_NULL(a_param)

        {
            CGRAPH_PARAM_WRITE_CODE_BLOCK(a_param)
            a_param->results.push_back(res_);
        }
        return CStatus();
    }
};

#endif //GRAPHANNS_C7_ROUTING_KGRAPH_H
