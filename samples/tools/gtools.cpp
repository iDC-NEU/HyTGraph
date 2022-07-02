// ----------------------------------------------------------------
// SEP-Graph: Finding Shortest Execution Paths for Graph Processing under a Hybrid Framework on GPU
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE
// in the root directory of this source distribution.
// ----------------------------------------------------------------
#include <algorithm>
#include <vector>
#include <set>
#include <map>
#include <gflags/gflags.h>
#include <utils/utils.h>
#include "gtools.h"

DEFINE_string(graphfile, "", "A file with a graph in Dimacs 10 format");
DEFINE_bool(ggr, true, "Graph file is a Galois binary GR file");
DEFINE_string(output, "", "edge list output");
DEFINE_string(out_degree, "", "Save out degree to file");
DEFINE_string(splitter, "\t", "the splitter between nodes pair");
DEFINE_bool(deself_cycle, false, "remove self-cycle, e.g. 0-0, 1-1 ...");
DEFINE_bool(deduplicate, true, "remove duplicated edges");
DEFINE_string(format, "gr", "graph format (gr/market/metis formats are supported)");
DEFINE_bool(d2ud, false, "convert directed graph to undirected graph");

int main(int argc, char **argv) {

    return 0;
}