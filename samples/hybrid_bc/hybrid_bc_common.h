// ----------------------------------------------------------------
// SEP-Graph: Finding Shortest Execution Paths for Graph Processing under a Hybrid Framework on GPU
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE
// in the root directory of this source distribution.
// ----------------------------------------------------------------
#ifndef HYBRID_BC_COMMON_H
#define HYBRID_BC_COMMON_H

#include <climits>
#include <gflags/gflags.h>
#include <groute/graphs/csr_graph.h>

typedef uint32_t level_t;
typedef float centrality_t;
typedef float sigma_t;

#define IDENTITY_ELEMENT UINT32_MAX
#define ERROR_THRESHOLD 0.05

std::pair<std::vector<centrality_t>, std::vector<sigma_t >>
BetweennessCentralityHost(const groute::graphs::host::CSRGraph &graph, index_t src);

int BCCheckErrors(std::vector<float> &regression_bc_values, std::vector<float> &bc_values);

int BCOutput(const char *file, const std::vector<centrality_t> &bc_values);

#endif // HYBRID_BC_COMMON_H
