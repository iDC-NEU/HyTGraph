// ----------------------------------------------------------------
// SEP-Graph: Finding Shortest Execution Paths for Graph Processing under a Hybrid Framework on GPU
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE
// in the root directory of this source distribution.
// ----------------------------------------------------------------
#ifndef HYBRID_ASYNC_PUSH_DD_H
#define HYBRID_ASYNC_PUSH_DD_H

#include <utils/cuda_utils.h>
#include <groute/device/cta_scheduler_hybrid.cuh>
#include <framework/variants/push_functor.h>
#include <framework/variants/common.cuh>
#include <framework/common.h>

namespace sepgraph
{
    namespace kernel
    {
        namespace async_push_dd
        {
            using sepgraph::common::LoadBalancing;

            template<typename TAppInst,
                    typename WorkSource,
                    typename WorkTarget,
                    typename CSRGraph,
                    template<typename> class GraphDatum,
                    typename TValue,
                    typename TBuffer,
                    typename TWeight>
            __forceinline__ __device__
            void Relax(TAppInst app_inst,
                       WorkSource work_source,
                       WorkTarget work_target_low,
                       WorkTarget work_target_high,
                       TBuffer current_priority,
                       CSRGraph csr_graph,
                       GraphDatum<TValue> node_value_datum,
                       GraphDatum<TBuffer> node_buffer_datum,
                       GraphDatum<TWeight> edge_weight_datum)
            {
                uint32_t tid = TID_1D;
                uint32_t nthreads = TOTAL_THREADS_1D;
                uint32_t work_size = work_source.get_size();
                PushFunctor<TAppInst, CSRGraph, GraphDatum, TBuffer, TWeight>
                        push_functor(app_inst,
                                     work_target_low,
                                     work_target_high,
                                     current_priority,
                                     csr_graph,
                                     node_buffer_datum,
                                     edge_weight_datum);

                for (int i = 0 + tid; i < work_size; i += nthreads)
                {
                    index_t node = work_source.get_work(i);
                    auto pair = app_inst.CombineValueBuffer(node,
                                                            node_value_datum.get_item_ptr(node),
                                                            node_buffer_datum.get_item_ptr(node));

                    if (pair.second)
                    {
                        Payload<TBuffer> payload;
                        payload.m_src = node;
                        payload.m_buffer_to_push = pair.first;

                        for (index_t edge = csr_graph.begin_edge(node), end_edge = csr_graph.end_edge(node); edge < end_edge; edge++)
                        {
                            if (!push_functor(edge, payload))
                            {
                                break;
                            }
                        }
                    }
                }
            }

            template<LoadBalancing LB,
                    typename TAppInst,
                    typename WorkSource,
                    typename WorkTarget,
                    typename CSRGraph,
                    template<typename> class GraphDatum,
                    typename TValue,
                    typename TBuffer,
                    typename TWeight>
            __forceinline__ __device__
            void RelaxCTA(TAppInst app_inst,
                          WorkSource work_source,
                          WorkTarget work_target_low,
                          WorkTarget work_target_high,
                          TBuffer current_priority,
                          CSRGraph csr_graph,
                          GraphDatum<TValue> node_value_datum,
                          GraphDatum<TBuffer> node_buffer_datum,
                          GraphDatum<TWeight> edge_weight_datum)
            {
                const uint32_t tid = TID_1D;
                const uint32_t nthreads = TOTAL_THREADS_1D;
                const uint32_t work_size = work_source.get_size();
                const uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;
                PushFunctor<TAppInst, CSRGraph, GraphDatum, TBuffer, TWeight>
                        push_functor(app_inst, work_target_low, work_target_high, current_priority,
                                     csr_graph, node_buffer_datum, edge_weight_datum);

                for (int i = 0 + tid; i < work_size_rup; i += nthreads)
                {
                    groute::dev::np_local<Payload<TBuffer>> np_local = {0, 0};

                    if (i < work_size)
                    {
                        const index_t node = work_source.get_work(i);
                        auto pair = app_inst.CombineValueBuffer(node,
                                                                node_value_datum.get_item_ptr(node),
                                                                node_buffer_datum.get_item_ptr(node));

                        if (pair.second)
                        {
                            np_local.start = csr_graph.begin_edge(node);
                            np_local.size = csr_graph.end_edge(node) - np_local.start; // out-degree
                            Payload<TBuffer> payload;
                            payload.m_src = node;
                            payload.m_buffer_to_push = pair.first;
                            np_local.meta_data = payload;
                        }
                    }

                    switch (LB)
                    {
                        case LoadBalancing::COARSE_GRAINED:
                            groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_COARSE_GRAINED>::template
                            schedule(np_local, push_functor);
                            break;
                        case LoadBalancing::FINE_GRAINED:
                            groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_FINE_GRAINED>::template
                            schedule(np_local, push_functor);
                            break;
                        case LoadBalancing::HYBRID:
                            groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_HYBRID>::template
                            schedule(np_local, push_functor);
                            break;
                        default:
                            assert(false);
                    }
                }
            }
        }
    }
}
#endif //HYBRID_ASYNC_PUSH_DD_H
