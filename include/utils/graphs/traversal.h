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

#ifndef __GRAPHS_TRAVERSAL_H
#define __GRAPHS_TRAVERSAL_H

#include <vector>
#include <map>
#include <algorithm>
#include <random>
#include <cassert>
#include <sstream>

#include <groute/event_pool.h>
#include <groute/graphs/csr_graph.h>
//#include <groute/dwl/distributed_worklist.cuh>

#include <utils/parser.h>
#include <utils/utils.h>
#include <utils/stopwatch.h>
#include <utils/markers.h>

#include <gflags/gflags.h>

DECLARE_int32(weight_num);
DECLARE_int32(SEGMENT);
DECLARE_string(output);
DECLARE_bool(check);
DECLARE_bool(verbose);
DECLARE_bool(trace);

DECLARE_string(graphfile);
DECLARE_string(format);

DECLARE_bool(gen_graph);
DECLARE_int32(gen_nnodes);
DECLARE_int32(gen_factor);
DECLARE_int32(gen_method);
DECLARE_bool(gen_weights);
DECLARE_int32(gen_weight_range);

DECLARE_int32(block_size);
DECLARE_int32(prio_delta);
DECLARE_bool(stats);

using std::min;
using std::max;

inline void KernelSizing(dim3 &grid_dims, dim3 &block_dims, uint32_t work_size)
{
    dim3 bd(FLAGS_block_size, 1, 1);
    //index_t numblocks_kernel = ((work_size * 32 + FLAGS_block_size) / FLAGS_block_size);
    //dim3 gd(FLAGS_block_size, ( numblocks_kernel+FLAGS_block_size)/FLAGS_block_size);
    dim3 gd(round_up(work_size, bd.x), 1, 1);

    if (grid_dims.x > 480)grid_dims.x = 480;
    grid_dims = gd;
    block_dims = bd;
}

namespace utils
{
    namespace traversal
    {
        /*
        * @brief A global context for graph traversal workers
        */
        template<typename Algo>
        class Context : public groute::Context
        {
        public:
            groute::graphs::host::CSRGraph host_graph;

            int ngpus;
            int nvtxs;
            uint64_t nedges;
            index_t segment = FLAGS_SEGMENT;
            index_t segment_ct;
	        uint64_t seg_nedge_csr[512];
	        uint64_t seg_nedge_csc[512];
	        uint64_t seg_sedge_csr[512];
	        uint64_t seg_sedge_csc[512];
            uint64_t seg_nedge_csr_ct[512];
            uint64_t seg_sedge_csr_ct[512];
	        index_t seg_snode[512];
	        index_t seg_enode[512];
            index_t seg_snode_ct[512];
            index_t seg_enode_ct[512];
            index_t segment_id_ct[512];

            Context(int ngpus=1) :
                    groute::Context(), ngpus(ngpus)
            {
                if (FLAGS_gen_graph) //Judge whether to generate the graph by yourself
                {
                    printf("\nGenerating graph, nnodes: %d, gen_factor: %d", FLAGS_gen_nnodes, FLAGS_gen_factor);

                    switch (FLAGS_gen_method)
                    {
                        case 1: // No intersection chain 
                        {
                            groute::graphs::host::NoIntersectionGraphGenerator generator(ngpus, FLAGS_gen_nnodes, FLAGS_gen_factor);
                            generator.Gen(host_graph);
                        }
                            break;
                        case 2: // Chain 
                        {
                            groute::graphs::host::ChainGraphGenerator generator(ngpus, FLAGS_gen_nnodes, FLAGS_gen_factor);
                            generator.Gen(host_graph);
                        }
                            break;
                        case 3: // Full cliques no intersection 
                        {
                            groute::graphs::host::CliquesNoIntersectionGraphGenerator generator(ngpus, FLAGS_gen_nnodes, FLAGS_gen_factor);
                            generator.Gen(host_graph);
                        }
                            break;
                        default:
                        {
                            // Generates a simple random graph with 'nnodes' nodes and maximum 'gen_factor' neighbors
                            groute::graphs::host::CSRGraphGenerator generator(FLAGS_gen_nnodes, FLAGS_gen_factor);
                            generator.Gen(host_graph);
                        }
                            break;
                    }
                }

                else
                {
                    graph_t *graph;

                    if (FLAGS_graphfile == "")
                    {
                        printf("A Graph File must be provided\n");
                        exit(0);
                    }

                    printf("\nLoading graph %s (%s)\n", FLAGS_graphfile.substr(FLAGS_graphfile.find_last_of('\\') + 1).c_str(), FLAGS_format.c_str());
                    graph = GetCachedGraph(FLAGS_graphfile, FLAGS_format,FLAGS_weight_num);

                    if (graph->nvtxs == 0)
                    {
                        printf("Empty graph!\n");
                        exit(0);
                    }

                    host_graph.Bind(
                            graph->nvtxs, graph->nedges,
                            graph->xadj, graph->adjncy,
                            graph->readew ? graph->adjwgt : nullptr,
                            graph->readvw ? graph->vwgt : nullptr // avoid binding to the default 1's weight arrays
                    );
		    
                    if (FLAGS_stats)
                    {
                        printf(
                                "The graph has %d vertices, and %ld edges (avg. degree: %f, max. degree: %d)\n",
                                host_graph.nnodes, host_graph.nedges, (float) host_graph.nedges / host_graph.nnodes, host_graph.max_degree());

                        CleanupGraphs();
                        exit(0);
                    }
                }

                if (host_graph.edge_weights == nullptr && FLAGS_gen_weights)
                {

                    if (FLAGS_verbose)
                        printf("\nNo edge data in the input graph, generating edge weights from the range [%d, %d]\n", 1, FLAGS_gen_weight_range);

                    // Generate edge data
                    std::default_random_engine generator;
                    std::uniform_int_distribution<int> distribution(1, FLAGS_gen_weight_range);

                    host_graph.AllocWeights();

                    for (int i = 0; i < host_graph.nedges; i++)
                    {
                        host_graph.edge_weights[i] = distribution(generator);
                    }
                }

                nvtxs = host_graph.nnodes;
                nedges = host_graph.nedges;

                printf("\n----- Running %s -----\n\n", Algo::Name());

                if (FLAGS_verbose)
                {
                    printf("The graph has %d vertices, and %ld edges (average degree: %f,max. degree: %d)\n", nvtxs, nedges, (float) nedges / nvtxs, host_graph.max_degree());
                }
            }
        };

    }
}

#endif // __GRAPHS_TRAVERSAL_H
