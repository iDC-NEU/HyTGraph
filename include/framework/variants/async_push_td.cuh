// ----------------------------------------------------------------
// SEP-Graph: Finding Shortest Execution Paths for Graph Processing under a Hybrid Framework on GPU
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE
// in the root directory of this source distribution.
// ----------------------------------------------------------------
#ifndef HYBRID_ASYNC_PUSH_TD_H
#define HYBRID_ASYNC_PUSH_TD_H

#include <cub/cub.cuh>
#include <framework/variants/push_functor.h>
#include <framework/variants/common.cuh>
#include <groute/graphs/csr_graph.h>
#include <groute/device/queue.cuh>
#include <groute/device/cta_scheduler_hybrid.cuh>

namespace sepgraph
{
    namespace kernel
    {
        namespace async_push_td
        {
            using sepgraph::common::LoadBalancing;

            template<bool enable_priority,
                    typename TAppInst,
                    typename WorkSource,
                    typename CSRGraph,
                    template<typename> class GraphDatum,
                    typename TValue,
                    typename TBuffer,
                    typename TWeight>
            __forceinline__ __device__
            void Relax(TAppInst app_inst,
                       WorkSource work_source,
                       CSRGraph csr_graph,
                       GraphDatum<TValue> node_value_datum,
                       GraphDatum<TBuffer> node_buffer_datum,
                       GraphDatum<TWeight> edge_weight_datum,
                       TBuffer current_priority)
            {
                uint32_t tid = TID_1D;
                uint32_t nthreads = TOTAL_THREADS_1D;
                uint32_t work_size = work_source.get_size();
                PushFunctor<TAppInst, CSRGraph, GraphDatum, TBuffer, TWeight>
                        push_functor(app_inst, csr_graph, node_buffer_datum, edge_weight_datum);

                for (uint32_t i = 0 + tid; i < work_size; i += nthreads)
                {
                    index_t node = work_source.get_work(i);

                    if (!enable_priority || app_inst.IsHighPriority(current_priority, node_buffer_datum.get_item(node)))
                    {
                        auto pair = app_inst.CombineValueBuffer(node,
                                                                node_value_datum.get_item_ptr(node),
                                                                node_buffer_datum.get_item_ptr(node));

                        // Value changed means validate combine, we need push the buffer to neighbors
                        if (pair.second)
                        {
                            index_t begin_edge = csr_graph.begin_edge(node),
                                    end_edge = csr_graph.end_edge(node);
                            Payload<TBuffer> payload;
                            payload.m_src = node;
                            payload.m_buffer_to_push = pair.first;

                            for (index_t edge = begin_edge; edge < end_edge; edge++)
                            {
                                push_functor(edge, payload);
                            }
                        }
                    }
                }
            }


            template<LoadBalancing LB,
                    bool enable_priority,
                    typename TAppInst,
                    typename WorkSource,
                    typename CSRGraph,
                    template<typename> class GraphDatum,
                    typename TValue,
                    typename TBuffer,
                    typename TWeight>
            __forceinline__ __device__
            void RelaxCTA(TAppInst app_inst,
                          WorkSource work_source,
                          CSRGraph csr_graph,
                          GraphDatum<TValue> node_value_datum,
                          GraphDatum<TBuffer> node_buffer_datum,
                          GraphDatum<TWeight> edge_weight_datum,
                          TBuffer current_priority)
            {
                const uint32_t tid = TID_1D;
                const uint32_t nthreads = TOTAL_THREADS_1D;
                const uint32_t work_size = work_source.get_size();
                const uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;
                PushFunctor<TAppInst, CSRGraph, GraphDatum, TBuffer, TWeight>
                        push_functor(app_inst, csr_graph, node_buffer_datum, edge_weight_datum);

                for (uint32_t i = 0 + tid; i < work_size_rup; i += nthreads)
                {
                    groute::dev::np_local<Payload<TBuffer>> np_local = {0, 0};

                    if (i < work_size)
                    {
                        const index_t node = work_source.get_work(i);

                        if (!enable_priority || app_inst.IsHighPriority(current_priority, node_buffer_datum.get_item(node)))
                        {
                            const auto pair = app_inst.CombineValueBuffer(node,
                                                                          node_value_datum.get_item_ptr(node),
                                                                          node_buffer_datum.get_item_ptr(node));

                            // Value changed means validate combine, we need push the buffer to neighbors
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

#endif //HYBRID_ASYNC_PUSH_TD_H
