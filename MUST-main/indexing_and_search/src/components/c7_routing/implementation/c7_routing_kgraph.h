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
        if (Params.entry_strategy_ == 1 || Params.entry_strategy_ == 2) {
            const auto &centroids1 = model_->centroids_meta_modal1_;
            const auto &centroids2 = model_->centroids_meta_modal2_;
            if (centroids1.data != nullptr && centroids1.num > 0 &&
                (Params.entry_strategy_ == 1 ||
                 (centroids2.data != nullptr && centroids2.num > 0))) {
                IDType best_centroid = 0;
                DistResType best_dist = std::numeric_limits<DistResType>::max();
                const VecValType1 *q1 = query_modal1_ + ((size_t)query_id_ * (size_t)dim1_);
                const VecValType2 *q2 = query_modal2_ + ((size_t)query_id_ * (size_t)dim2_);
                for (IDType i = 0; i < centroids1.num; ++i) {
                    DistResType1 d1 = 0;
                    DistResType2 d2 = 0;
                    DistResType dist = 0;
                    const VecValType1 *c1 = centroids1.data + (size_t)i * (size_t)centroids1.dim;
                    if (Params.entry_strategy_ == 1) {
                        dist_op_.dist_op1_.calculate(q1, c1, dim1_, centroids1.dim, d1);
                        dist = d1 * dist_op_.weight_1_;
                    } else {
                        if (dist_op_.weight_1_ != 0) {
                            dist_op_.dist_op1_.calculate(q1, c1, dim1_, centroids1.dim, d1);
                        }
                        const VecValType2 *c2 = centroids2.data + (size_t)i * (size_t)centroids2.dim;
                        if (dist_op_.weight_2_ != 0) {
                            dist_op_.dist_op2_.calculate(q2, c2, dim2_, centroids2.dim, d2);
                        }
                        dist = d1 * dist_op_.weight_1_ + d2 * dist_op_.weight_2_;
                    }
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_centroid = i;
                    }
                }
                IDType mapped_id = best_centroid;
                if (model_->centroid_ids_meta_.data != nullptr &&
                    model_->centroid_ids_meta_.num > best_centroid &&
                    model_->centroid_ids_meta_.dim > 0) {
                    mapped_id = model_->centroid_ids_meta_.data[
                        (size_t)best_centroid * (size_t)model_->centroid_ids_meta_.dim];
                }
                if (mapped_id < num_) {
                    best_ep = mapped_id;
                    DistResType ep_dist = 0;
                    dist_op_.calculate(q1,
                                      data_modal1_ + (size_t)best_ep * (size_t)dim1_,
                                      dim1_, dim1_,
                                      q2,
                                      data_modal2_ + (size_t)best_ep * (size_t)dim2_,
                                      dim2_, dim2_, ep_dist);
                    s_param->sp.assign(search_L_ + 1, NeighborFlag(0,
                                                                  std::numeric_limits<DistResType>::max(),
                                                                  false));
                    s_param->sp[0] = NeighborFlag(best_ep, ep_dist, true);
                    std::sort(s_param->sp.begin(), s_param->sp.begin() + search_L_);
                }
            }
        }
        std::cout << "[DEBUG] Strategy: " << Params.entry_strategy_
                  << ", Selected Entry Point ID: " << best_ep << std::endl;

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
