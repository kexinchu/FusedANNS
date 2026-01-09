/***************************
@Author: xxx
@Contact: xxx@xxx.com
@File: T-siftsmall.cpp
@Time: 2022/4/30 17:24
@Desc:
***************************/

#include "src/graph_anns.h"
#include <string>
#include <vector>

using namespace CGraph;

namespace {
struct ParsedArgs {
    std::vector<char*> positional;
    bool has_norm_modal1 = false;
    bool has_norm_modal2 = false;
    unsigned norm_modal1 = 0;
    unsigned norm_modal2 = 0;
};

bool parse_args(int argc, char **argv, ParsedArgs& out, std::string& error) {
    out.positional.clear();
    out.positional.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
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

void print_usage(const char* exe) {
    std::cout << "[RUN] Usage: " << exe
              << " <modal1_base> <modal2_base> <index_path> <thread_num>"
              << " <w1> <w2> <L_candidate> <R_neighbor> <C_neighbor> <k_init_graph>"
              << " <nn_size> <rnn_size> <pool_size> <iter> <sample_num>"
              << " <graph_quality_threshold> <is_norm_modal1> <is_norm_modal2>"
              << " <is_skip> <skip_num>"
              << " [--norm_modal1 0/1] [--norm_modal2 0/1]" << std::endl;
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

    unsigned is_norm_modal1 = strtoul(args[17], nullptr, 10);
    unsigned is_norm_modal2 = strtoul(args[18], nullptr, 10);
    if (parsed.has_norm_modal1) {
        is_norm_modal1 = parsed.norm_modal1;
    }
    if (parsed.has_norm_modal2) {
        is_norm_modal2 = parsed.norm_modal2;
    }

    std::cout << "[RUN] Execution object: " << args[0] << std::endl;         // 0
    std::cout << "[PARAM] Modal1 base path: " << args[1] << std::endl;       // 1
    std::cout << "[PARAM] Modal2 base path: " << args[2] << std::endl;       // 2
    std::cout << "[PARAM] Index path: " << args[3] << std::endl;             // 3
    std::cout << "[PARAM] thread number: " << args[4] << std::endl;          // 4
    std::cout << "[PARAM] Modal1 distance weight: " << args[5] << std::endl; // 5
    std::cout << "[PARAM] Modal2 distance weight: " << args[6] << std::endl; // 6
    std::cout << "[PARAM] L_candidate: " << args[7] << std::endl;            // 7
    std::cout << "[PARAM] R_neighbor: " << args[8] << std::endl;             // 8
    std::cout << "[PARAM] C_neighbor: " << args[9] << std::endl;             // 9
    std::cout << "[PARAM] k_init_graph: " << args[10] << std::endl;          // 10
    std::cout << "[PARAM] nn_size: " << args[11] << std::endl;               // 11
    std::cout << "[PARAM] rnn_size: " << args[12] << std::endl;              // 12
    std::cout << "[PARAM] pool_size: " << args[13] << std::endl;             // 13
    std::cout << "[PARAM] iter time: " << args[14] << std::endl;             // 14
    std::cout << "[PARAM] sample number: " << args[15] << std::endl;         // 15
    std::cout << "[PARAM] graph quality threshold: " << args[16] << std::endl;// 16
    std::cout << "[PARAM] is norm for modal1?: " << is_norm_modal1 << std::endl;    // 17
    std::cout << "[PARAM] is norm for modal2?: " << is_norm_modal2 << std::endl;    // 18
    std::cout << "[PARAM] is skip number for modal2?: " << args[19] << std::endl;    // 19
    std::cout << "[PARAM] skip number for modal2?: " << args[20] << std::endl;    // 20
    unsigned thread_num = strtoul(args[4], nullptr, 10);
    unsigned is_skip = strtoul(args[19], nullptr, 10);
    unsigned skip_num = strtoul(args[20], nullptr, 10);
    Params.set_general_param(thread_num, is_norm_modal1, is_norm_modal2, is_skip, skip_num);
    float w1 = strtof(args[5], nullptr);
    float w2 = strtof(args[6], nullptr);
    Params.set_data_param(w1, w2);
    unsigned L_candidate = strtoul(args[7], nullptr, 10);
    unsigned R_neighbor = strtoul(args[8], nullptr, 10);
    unsigned C_neighbor = strtoul(args[9], nullptr, 10);
    unsigned k_init_graph = strtoul(args[10], nullptr, 10);
    unsigned nn_size = strtoul(args[11], nullptr, 10);
    unsigned rnn_size = strtoul(args[12], nullptr, 10);
    unsigned pool_size = strtoul(args[13], nullptr, 10);
    unsigned iter = strtoul(args[14], nullptr, 10);
    unsigned sample_num = strtoul(args[15], nullptr, 10);
    float threshold = strtof(args[16], nullptr);
    Params.set_train_param(args[1], args[2], args[3],
                           L_candidate, R_neighbor, C_neighbor, k_init_graph, nn_size, rnn_size, pool_size,
                           iter, sample_num, threshold);

    GPipelinePtr pipeline = GPipelineFactory::create();
    GElementPtr a, b, c, d, e, cde_region, f = nullptr;

    // build
    CStatus status = pipeline->registerGElement<ConfigAlgNPGNode, -1>(&a, {}, "config_npg");
    status += pipeline->registerGElement<ConfigModelNode, -2>(&b, {a}, "config_model");

    c = pipeline->createGNode<C1InitializationNNDescent>(GNodeInfo("c1_nssg"));
    d = pipeline->createGNode<C2CandidateNSSGV1>(GNodeInfo("c2_nssg"));
    e = pipeline->createGNode<C3NeighborNSGV1>(GNodeInfo("c3_nsg"));

    cde_region = pipeline->createGGroup<GCluster>({c, d, e});

    status += pipeline->registerGElement<GCluster>(&cde_region, {b}, "build");
    status += pipeline->registerGElement<SaveIndexNode>(&f, {cde_region}, "save_index");

    pipeline->addGAspect<TimerAspect>()->addGAspect<TraceAspect>();

    status += pipeline->process();
    if (!status.isOK()) {
        CGRAPH_ECHO("process graph error, error info is [%s]", status.getInfo().c_str());
        return 0;
    }
    GPipelineFactory::remove(pipeline);
    return 0;
}
