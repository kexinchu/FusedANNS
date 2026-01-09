/***************************
@Author: xxx
@Contact: xxx@xxx.com
@File: eva_recall_node.h
@Time: 2022/4/8 4:31 PM
@Desc: calculate recall rate
***************************/

#ifndef GRAPHANNS_EVA_RECALL_NODE_H
#define GRAPHANNS_EVA_RECALL_NODE_H

#include <algorithm>
#include <fstream>
#include <iterator>
#include <string>
#include "../../elements_define.h"

class EvaRecallNode : public CGraph::GNode {
public:
    CStatus init() override {

        auto m_param = CGRAPH_GET_GPARAM(AnnsModelParam, GA_ALG_MODEL_PARAM_KEY)
        CGRAPH_ASSERT_NOT_NULL(m_param)
        CStatus status = m_param->eva_meta_.load(Params.GA_ALG_GROUND_TRUTH_PATH_, 0);
        if (!status.isOK()) {
            CGRAPH_RETURN_ERROR_STATUS("EvaRecallNode init load param failed")
        }
        printf("[PARAM] gt num: %d\n", m_param->eva_meta_.num);
        printf("[PARAM] gt[0] dim: %d\n", (unsigned)m_param->eva_meta_.vec[0].size());

        gt_num_ = m_param->eva_meta_.num;
        return status;
    }

    CStatus run() override {
        auto *m_param = CGRAPH_GET_GPARAM(AnnsModelParam, GA_ALG_MODEL_PARAM_KEY)
        auto *s_param = CGRAPH_GET_GPARAM(AlgParamBasic, GA_ALG_PARAM_BASIC_KEY);
        if (nullptr == m_param || nullptr == s_param) {
            CGRAPH_RETURN_ERROR_STATUS("EvaRecallNode run get param failed")
        }

        unsigned top_k = s_param->top_k;
        unsigned gt_k = Params.gt_k_;
        int cnt = 0;
        std::set<IDType> gt, res;
        const std::string& per_query_path = Params.GA_ALG_PER_QUERY_PATH_;
        std::ofstream per_query_out;
        const bool write_per_query = !per_query_path.empty() && per_query_path != " ";
        if (write_per_query) {
            per_query_out.open(per_query_path);
            if (!per_query_out.is_open()) {
                CGRAPH_RETURN_ERROR_STATUS("EvaRecallNode open per-query result file failed")
            }
        }
        for (unsigned i = 0; i < gt_num_; i++) {
            const size_t gt_size = m_param->eva_meta_.vec[i].size();
            gt_look_num_ = (Params.is_multi_res_equal_ ? gt_size : gt_k);
            if (gt_look_num_ > gt_size) {
                gt_look_num_ = gt_size;
            }
            gt.clear();
            res.clear();
            gt.insert(m_param->eva_meta_.vec[i].begin(), m_param->eva_meta_.vec[i].begin() + gt_look_num_);
            const size_t res_size = std::min<size_t>(s_param->results[i].size(), top_k);
            res.insert(s_param->results[i].begin(), s_param->results[i].begin() + res_size);
            std::vector<IDType> res_intersection;
            std::set_intersection(gt.begin(), gt.end(), res.begin(), res.end(),
                                  std::insert_iterator<std::vector<IDType>>(res_intersection,
                                          res_intersection.begin()));
            cnt += (res_intersection.size() >= gt_k ? (int) gt_k : (int) res_intersection.size());

            if (write_per_query) {
                long long retrieved_id = -1;
                long long gt_id = -1;
                bool is_correct = false;
                if (!s_param->results[i].empty()) {
                    retrieved_id = static_cast<long long>(s_param->results[i][0]);
                    is_correct = gt.find(s_param->results[i][0]) != gt.end();
                }
                if (!m_param->eva_meta_.vec[i].empty()) {
                    gt_id = static_cast<long long>(m_param->eva_meta_.vec[i][0]);
                }
                per_query_out << i << ", " << (is_correct ? 1 : 0) << ", "
                              << retrieved_id << ", " << gt_id << "\n";
            }
        }

        float acc = (float) cnt / (float) (gt_num_ * gt_k);
        printf("[EVA] %d NN accuracy for top%d: %f\n", gt_k, top_k, acc);
        return CStatus();
    }

private:
    unsigned gt_num_ = 0;
    unsigned gt_look_num_ = 0;
};

#endif //GRAPHANNS_EVA_RECALL_NODE_H
