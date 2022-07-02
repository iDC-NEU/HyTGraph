// ----------------------------------------------------------------
// SEP-Graph: Finding Shortest Execution Paths for Graph Processing under a Hybrid Framework on GPU
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE
// in the root directory of this source distribution.
// ----------------------------------------------------------------
#include "hybrid_bc_common.h"
#include <utils/stopwatch.h>
#include <unordered_set>
#include <queue>

DEFINE_int32(src, 0, "The source node of BC algorithm");
DECLARE_bool(verbose);


std::pair<std::vector<centrality_t>, std::vector<sigma_t >>
BetweennessCentralityHost(const groute::graphs::host::CSRGraph &graph,
                          index_t src) {
    Stopwatch sw(true);

    std::vector<int> source_path(graph.nnodes, -1);
    std::vector<centrality_t> bc_values(graph.nnodes, 0.0);
    std::vector<sigma_t> sigmas(graph.nnodes, 0.0);

    source_path[src] = 0;
    int search_depth = 0;
    sigmas[src] = 1;

    std::queue<index_t> wl1, wl2;
    std::queue<index_t> *in_wl = &wl1, *out_wl = &wl2;

    in_wl->push(src);


    while (!in_wl->empty()) {
        while (!in_wl->empty()) {
            index_t node = in_wl->front();
            in_wl->pop();

            int nbr_dist = source_path[node] + 1;

            for (index_t edge = graph.begin_edge(node),
                         end_edge = graph.end_edge(node); edge < end_edge; edge++) {
                index_t dest = graph.edge_dest(edge);

                if (source_path[dest] == -1) {
                    source_path[dest] = nbr_dist;
                    sigmas[dest] += sigmas[node];

                    if (search_depth < nbr_dist) {
                        search_depth = nbr_dist;
                    }

                    out_wl->push(dest);
                } else {
                    if (source_path[dest] == source_path[node] + 1) {
                        sigmas[dest] += sigmas[node];
                    }
                }
            }
        }
        std::swap(in_wl, out_wl);
    }
    search_depth++;

    for (int iter = search_depth - 2; iter > 0; --iter) {
        for (index_t node = 0; node < graph.nnodes; node++) {
            if (source_path[node] == iter) {
                for (index_t edge = graph.begin_edge(node),
                             end_edge = graph.end_edge(node); edge < end_edge; edge++) {
                    index_t dest = graph.edge_dest(edge);

                    if (source_path[dest] == iter + 1) {
                        bc_values[node] += 1.0f * sigmas[node] / sigmas[dest] *
                                           (1.0f + bc_values[dest]);
                    }
                }
            }
        }
    }

    for (index_t node = 0; node < graph.nnodes; node++) {
        bc_values[node] *= 0.5f;
    }
    sw.stop();

    if (FLAGS_verbose) {
        printf("\nBC Host: %f ms. \n", sw.ms());
    }

    return std::make_pair(bc_values, sigmas);
}

int BCCheckErrors(std::vector<float> &regression_bc_values, std::vector<float> &bc_values) {
    if (bc_values.size() != regression_bc_values.size()) {
        return std::abs((int64_t) bc_values.size() - (int64_t) regression_bc_values.size());
    }

    index_t nnodes = bc_values.size();
    int total_errors = 0;

    for (index_t node = 0; node < nnodes; node++) {

        bool is_right = true;

        if (fabs(bc_values[node] - 0.0) < 0.01f) {
            if (fabs(bc_values[node] - regression_bc_values[node]) > ERROR_THRESHOLD) {
                is_right = false;
            }
        } else {
            if (fabs((bc_values[node] - regression_bc_values[node]) / regression_bc_values[node]) > ERROR_THRESHOLD) {
                is_right = false;
            }
        }

        if (!is_right) {
            fprintf(stderr, "node: %u bc regression: %f device: %f\n", node,
                    regression_bc_values[node],
                    bc_values[node]);
            total_errors++;
        }
    }

    printf("Total errors: %u\n", total_errors);

    return total_errors;
}

int BCOutput(const char *file, const std::vector<centrality_t> &bc_values) {
    FILE *f;
    f = fopen(file, "w");

    if (f) {
        for (int i = 0; i < bc_values.size(); ++i) {
            fprintf(f, "%u %f\n", i, bc_values[i]);
        }
        fclose(f);

        return 1;
    } else {
        fprintf(stderr, "Could not open '%s' for writing\n", file);
        return 0;
    }
}