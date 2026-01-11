/***************************
@Author: xxx
@Contact: xxx@xxx.com
@File: S-siftsmall.cpp
@Time: 2022/4/30 17:25
@Desc:
***************************/

#include "src/graph_anns.h"
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

using namespace CGraph;

namespace {
struct ParsedArgs {
    std::vector<char*> positional;
    bool has_w1 = false;
    bool has_w2 = false;
    float w1 = 0.0f;
    float w2 = 0.0f;
    bool has_norm_modal1 = false;
    bool has_norm_modal2 = false;
    unsigned norm_modal1 = 0;
    unsigned norm_modal2 = 0;
    std::string per_query_path;
    bool has_entry_strategy = false;
    unsigned entry_strategy = 0;
    bool has_entry_topk = false;
    unsigned entry_topk = 1;
    std::string centroids_visual_path;
    std::string centroids_attr_path;
};

bool parse_args(int argc, char **argv, ParsedArgs& out, std::string& error) {
    out.positional.clear();
    out.positional.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--w1") {
            if (i + 1 >= argc) {
                error = "--w1 requires a value";
                return false;
            }
            out.has_w1 = true;
            out.w1 = strtof(argv[++i], nullptr);
            continue;
        }
        if (arg == "--w2") {
            if (i + 1 >= argc) {
                error = "--w2 requires a value";
                return false;
            }
            out.has_w2 = true;
            out.w2 = strtof(argv[++i], nullptr);
            continue;
        }
        if (arg == "--per_query_path" || arg == "--per-query-path") {
            if (i + 1 >= argc) {
                error = "--per_query_path requires a value";
                return false;
            }
            out.per_query_path = argv[++i];
            continue;
        }
        if (arg == "--entry_strategy" || arg == "--entry-strategy") {
            if (i + 1 >= argc) {
                error = "--entry_strategy requires a value";
                return false;
            }
            out.has_entry_strategy = true;
            out.entry_strategy = strtoul(argv[++i], nullptr, 10);
            continue;
        }
        if (arg == "--entry_topk" || arg == "--entry-topk") {
            if (i + 1 >= argc) {
                error = "--entry_topk requires a value";
                return false;
            }
            out.has_entry_topk = true;
            out.entry_topk = strtoul(argv[++i], nullptr, 10);
            continue;
        }
        if (arg == "--centroids_visual" || arg == "--centroids-visual") {
            if (i + 1 >= argc) {
                error = "--centroids_visual requires a value";
                return false;
            }
            out.centroids_visual_path = argv[++i];
            continue;
        }
        if (arg == "--centroids_attr" || arg == "--centroids-attr") {
            if (i + 1 >= argc) {
                error = "--centroids_attr requires a value";
                return false;
            }
            out.centroids_attr_path = argv[++i];
            continue;
        }
        if (arg == "--norm_modal1" || arg == "--norm-modal1") {
            if (i + 1 >= argc) {
                error = "--norm_modal1 requires a value";
                return false;
            }
            out.has_norm_modal1 = true;
            out.norm_modal1 = strtoul(argv[++i], nullptr, 10);
            continue;
        }
        if (arg == "--norm_modal2" || arg == "--norm-modal2") {
            if (i + 1 >= argc) {
                error = "--norm_modal2 requires a value";
                return false;
            }
            out.has_norm_modal2 = true;
            out.norm_modal2 = strtoul(argv[++i], nullptr, 10);
            continue;
        }
        if (!arg.empty() && arg.rfind("--", 0) == 0) {
            error = "Unknown flag: " + arg;
            return false;
        }
        out.positional.push_back(argv[i]);
    }
    return true;
}

std::string format_weight(float value) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4) << value;
    return oss.str();
}

void print_usage(const char* exe) {
    std::cout << "[RUN] Usage: " << exe
              << " <modal1_base> <modal2_base> <modal1_query> <modal2_query>"
              << " <groundtruth> <index_path> <save_result_path> <thread_num>"
              << " <w1> <w2> <top_k> <gt_k> <L_search>"
              << " <is_norm_modal1> <is_norm_modal2> <is_skip> <skip_num>"
              << " <is_multi_results_equal> <is_delete_id> <delete_id_path>"
              << " [--w1 val] [--w2 val] [--per_query_path path]"
              << " [--norm_modal1 0/1] [--norm_modal2 0/1]"
              << " [--entry_strategy 0/1/2] [--entry_topk K]"
              << " [--centroids_visual path]"
              << " [--centroids_attr path]" << std::endl;
}
} // namespace

int main(int argc, char **argv) {
    time_t tt = time(nullptr);
    tm* t=localtime(&tt);
    std::cout << std::endl;
    std::cout << t->tm_year + 1900 << "-" << t->tm_mon + 1 << "-" << t->tm_mday
              << "-" << t->tm_hour << "-" << t->tm_min << "-" << t->tm_sec << std::endl;
    std::cout << std::endl;

    ParsedArgs parsed;
    std::string parse_error;
    if (!parse_args(argc, argv, parsed, parse_error)) {
        if (!parse_error.empty()) {
            std::cout << "[RUN] " << parse_error << std::endl;
        }
        print_usage(argv[0]);
        return 1;
    }

    constexpr size_t kExpectedArgs = 21;
    if (parsed.positional.size() != kExpectedArgs) {
        print_usage(argv[0]);
        return 1;
    }

    auto args = parsed.positional;
    float w1 = strtof(args[9], nullptr);
    float w2 = strtof(args[10], nullptr);
    if (parsed.has_w1) {
        w1 = parsed.w1;
    }
    if (parsed.has_w2) {
        w2 = parsed.w2;
    }

    std::string per_query_path = parsed.per_query_path;
    std::string result_path(args[7]);
    if (per_query_path.empty() && !result_path.empty() && result_path != " ") {
        std::string base = result_path;
        const std::string ext = ".txt";
        if (base.size() >= ext.size() &&
            base.compare(base.size() - ext.size(), ext.size(), ext) == 0) {
            base = base.substr(0, base.size() - ext.size());
        }
        per_query_path = base + "_w1_" + format_weight(w1) + "_w2_" + format_weight(w2) + ".txt";
    }

    unsigned is_norm_modal1 = strtoul(args[14], nullptr, 10);
    unsigned is_norm_modal2 = strtoul(args[15], nullptr, 10);
    if (parsed.has_norm_modal1) {
        is_norm_modal1 = parsed.norm_modal1;
    }
    if (parsed.has_norm_modal2) {
        is_norm_modal2 = parsed.norm_modal2;
    }
    unsigned entry_strategy = parsed.entry_strategy;
    if (!parsed.has_entry_strategy) {
        entry_strategy = 0;
    }
    if (entry_strategy > 2) {
        std::cout << "[RUN] --entry_strategy must be 0, 1, or 2" << std::endl;
        return 1;
    }
    unsigned entry_topk = parsed.entry_topk;
    if (!parsed.has_entry_topk) {
        entry_topk = 1;
    }
    if (entry_topk == 0) {
        std::cout << "[RUN] --entry_topk must be >= 1" << std::endl;
        return 1;
    }

    std::cout << "[RUN] Execution object: " << args[0] << std::endl;         // 0
    std::cout << "[PARAM] Modal1 base path: " << args[1] << std::endl;       // 1
    std::cout << "[PARAM] Modal2 base path: " << args[2] << std::endl;       // 2
    std::cout << "[PARAM] Modal1 query path: " << args[3] << std::endl;      // 3
    std::cout << "[PARAM] Modal2 query path: " << args[4] << std::endl;      // 4
    std::cout << "[PARAM] Groundtruth path: " << args[5] << std::endl;       // 5
    std::cout << "[PARAM] Index path: " << args[6] << std::endl;             // 6
    std::cout << "[PARAM] Save Result path: " << args[7] << std::endl;       // 7
    std::cout << "[PARAM] thread number: " << args[8] << std::endl;          // 8
    std::cout << "[PARAM] Modal1 distance weight: " << w1 << std::endl;      // 9
    std::cout << "[PARAM] Modal2 distance weight: " << w2 << std::endl;      // 10
    std::cout << "[PARAM] top-k: " << args[11] << std::endl;                 // 11
    std::cout << "[PARAM] gt-k: " << args[12] << std::endl;                  // 12
    std::cout << "[PARAM] L_search: " << args[13] << std::endl;              // 13
    std::cout << "[PARAM] is norm for modal1?: " << is_norm_modal1 << std::endl;   // 14
    std::cout << "[PARAM] is norm for modal2?: " << is_norm_modal2 << std::endl;   // 15
    std::cout << "[PARAM] is skip number for modal2?: " << args[16] << std::endl;    // 16
    std::cout << "[PARAM] skip number for modal2?: " << args[17] << std::endl;    // 17
    std::cout << "[PARAM] is multi-results equal?: " << args[18] << std::endl;    // 18
    std::cout << "[PARAM] is delete id?: " << args[19] << std::endl;              // 19
    std::cout << "[PARAM] Delete id path: " << args[20] << std::endl;             // 20
    std::cout << "[PARAM] Entry strategy: " << entry_strategy << std::endl;
    std::cout << "[PARAM] Entry top-k: " << entry_topk << std::endl;
    if (!parsed.centroids_visual_path.empty()) {
        std::cout << "[PARAM] Centroids visual path: " << parsed.centroids_visual_path << std::endl;
    }
    if (!parsed.centroids_attr_path.empty()) {
        std::cout << "[PARAM] Centroids attr path: " << parsed.centroids_attr_path << std::endl;
    }
    if (!per_query_path.empty() && per_query_path != " ") {
        std::cout << "[PARAM] Per-query result path: " << per_query_path << std::endl;
    }

    unsigned top_k = strtoul(args[11], nullptr, 10);
    unsigned gtk = strtoul(args[12], nullptr, 10);
    unsigned l = strtoul(args[13], nullptr, 10);
    Params.set_search_param(args[1], args[2], args[3],
                            args[4], args[5], args[7], top_k, gtk,
                            l, args[6]);
    unsigned thread_num = strtoul(args[8], nullptr, 10);
    unsigned is_skip = strtoul(args[16], nullptr, 10);
    unsigned skip_num = strtoul(args[17], nullptr, 10);
    unsigned is_multiple_res_equal = strtoul(args[18], nullptr, 10);
    unsigned is_delete_id = strtoul(args[19], nullptr, 10);
    Params.set_general_param(thread_num, is_norm_modal1, is_norm_modal2, is_skip, skip_num,
                             is_multiple_res_equal, is_delete_id);
    Params.set_data_param(w1, w2);
    Params.set_entry_param(entry_strategy, entry_topk,
                           parsed.centroids_visual_path, parsed.centroids_attr_path);

    if (!per_query_path.empty() && per_query_path != " ") {
        Params.set_per_query_path(per_query_path);
    }
    if (is_delete_id) {
        Params.set_delete_id_path(args[20]);
    }

    GPipelinePtr pipeline = GPipelineFactory::create();

    GElementPtr a, b, f, g, h, i, p, gh_region= nullptr;
    // build
    CStatus status = pipeline->registerGElement<ConfigAlgNPGNode, -1>(&a, {}, "config_npg");
    status += pipeline->registerGElement<ConfigModelNode, -2>(&b, {a}, "config_model");

    status += pipeline->registerGElement<LoadIndexNode>(&f, {a}, "load_index");

    //search
    g = pipeline->createGNode<C6SeedKGraph>(GNodeInfo("c6_random"));
    h = pipeline->createGNode<C7RoutingKGraph>(GNodeInfo("c7_greedy"));

    gh_region = pipeline->createGGroup<SearchRegion>({g, h});
    status += pipeline->registerGElement<SearchRegion>(&gh_region, {f}, "search");

    status += pipeline->registerGElement<EvaRecallNode>(&i, {gh_region}, "eva_recall");

//    std::string result_path(argv[7]);
//    if (result_path != " ") {
//        status += pipeline->registerGElement<SaveResultNode>(&p, {i}, "save_result");
//    }

    gh_region->addGAspect<TimerAspect>()->addGAspect<TraceAspect>();
    status += pipeline->process();
    if (!status.isOK()) {
        CGRAPH_ECHO("process graph error, error info is [%s]", status.getInfo().c_str());
        return 0;
    }
    GPipelineFactory::remove(pipeline);

    return 0;
}
