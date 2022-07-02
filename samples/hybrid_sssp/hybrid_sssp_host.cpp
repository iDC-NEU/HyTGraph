// Groute: An Asynchronous Multi-GPU Programming Framework
// http://www.github.com/groute/groute
// Copyright (c) 2017, A. Barak
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the names of the copyright holders nor the names of its 
//   contributors may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <set>
#include <functional>

#include "hybrid_sssp_common.h"

std::vector<distance_t>
SSSPHostNaive(const groute::graphs::host::CSRGraph &graph, const distance_t *edge_weights, index_t source_node) {
    std::vector<distance_t> distances(graph.nnodes, IDENTITY_ELEMENT);

    std::queue<index_t> work;

    distances[source_node] = 0;
    work.push(source_node);

    while (!work.empty()) {
        index_t node = work.front();
        work.pop();

        distance_t distance = distances[node];
        for (index_t edge = graph.begin_edge(node), end_edge = graph.end_edge(node); edge < end_edge; ++edge) {
            index_t dest = graph.edge_dest(edge);
            distance_t edge_weight = edge_weights[edge];
            if (distances[dest] > distance + edge_weight) // if can be relaxed
            {
                distances[dest] = distance + edge_weight;
                work.push(dest);
            }
        }
    }

    return distances;
}

int SSSPCheckErrors(const std::vector<distance_t> &distances, const std::vector<distance_t> &regression) {
    if (distances.size() != regression.size()) {
        return std::abs((long long) distances.size() - (long long) regression.size());
    }

    int over_errors = 0, miss_errors = 0;
    std::vector<int> over_error_indices, miss_error_indices;

    distance_t
            max_over_delta = 0, max_miss_delta = 0;

    for (int i = 0; i < regression.size(); ++i) {
        distance_t hv = distances[i];
        distance_t rv = regression[i];

        if (hv > rv) {
	  //printf("id: %d dis:%d true_dis:%d \n",i,hv,rv);
            ++over_errors;
            over_error_indices.push_back(i);
            max_over_delta = std::max(max_over_delta, hv - rv);
        } else if (hv < rv) {
	  //printf("id: %d dis:%d true_dis:%d \n",i,hv,rv);
            ++miss_errors;
            miss_error_indices.push_back(i);
            max_miss_delta = std::max(max_miss_delta, rv - hv);
        }
    }

    if (miss_errors > 0)
        printf("Miss errors: %d\n\n", miss_errors);

    if (over_errors > 0)
        printf("Over errors: %d\n\n", over_errors);

    return (miss_errors + over_errors);
}

int SSSPOutput(const char *file, const std::vector<distance_t> &distances) {
    FILE *f;
    f = fopen(file, "w");

    if (f) {
        for (int i = 0; i < distances.size(); ++i) {
            if (distances[i] == IDENTITY_ELEMENT) {
                fprintf(f, "%u INF\n", i);
            } else {
                fprintf(f, "%u %u\n", i, distances[i]);
            }
        }
        fclose(f);

        return 1;
    } else {
        fprintf(stderr, "Could not open '%s' for writing\n", file);
        return 0;
    }
}
