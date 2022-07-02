// ----------------------------------------------------------------
// SEP-Graph: Finding Shortest Execution Paths for Graph Processing under a Hybrid Framework on GPU
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE
// in the root directory of this source distribution.
// ----------------------------------------------------------------
#ifndef HYBRID_ASYNC_PULL_TD_H
#define HYBRID_ASYNC_PULL_TD_H

#include <framework/variants/pull_functor.h>
#include <groute/device/cta_scheduler_hybrid.cuh>

namespace sepgraph
{
    namespace kernel
    {
        namespace async_pull_td
        {
            //TODO Is that possible using async pull for PR?
            //THINK AsyncPullTD PR?
            //Pull buffer from neighbor
            //barrier
            //CombineBuffer

            template<typename TAppInst,
                    typename WorkSource,
                    typename CSCGraph,
                    template<typename> class GraphDatum,
                    typename TBuffer,
                    typename TWeight>
            //__device__ __forceinline__
            __global__
            void Relax(TAppInst app_inst,
                       WorkSource work_source,
                       CSCGraph csc_graph,
                       GraphDatum<TBuffer> node_buffer_datum,
                       GraphDatum<TWeight> edge_weight_datum)
            {
//                const uint32_t tid = TID_1D;
//                const uint32_t nthreads = TOTAL_THREADS_1D;
//                const uint32_t work_size = node_buffer_datum.size;
//                PullFunctor <TAppInst, CSCGraph, GraphDatum, TBuffer, TWeight>
//                        pull_functor(app_inst, csc_graph, node_buffer_datum, edge_weight_datum);
//
//                for (uint32_t i = 0 + tid; i < work_size; i += nthreads)
//                {
//                    index_t dst = work_source.get_work(i);
//
//                    for (index_t edge = csc_graph.begin_edge(dst),
//                                 end_edge = csc_graph.end_edge(dst); edge < end_edge; edge++)
//                    {
//                        pull_functor(edge, dst);
//                    }
//                }
            }
        }
    }
}
#endif //HYBRID_ASYNC_PULL_TD_H
