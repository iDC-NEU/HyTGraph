// ----------------------------------------------------------------
// SEP-Graph: Finding Shortest Execution Paths for Graph Processing under a Hybrid Framework on GPU
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE
// in the root directory of this source distribution.
// ----------------------------------------------------------------
#ifndef HYBRID_API_H
#define HYBRID_API_H

#include <cuda.h>
#include <cub/grid/grid_barrier.cuh>
#include <groute/graphs/csr_graph.h>
//#include <groute/graphs/common.h>
#include <groute/device/queue.cuh>
#include <utils/cuda_utils.h>
#include <framework/graph_datum.cuh>

namespace sepgraph
{
    namespace api
    {
        template<typename TValue, typename TBuffer, typename TWeight>
        struct AppBase
        {
            // [ACCUMULATE] [CONTINUE/BREAK]
            const int ACCUMULATE_SUCCESS_BREAK = 0;         // 0 0
            const int ACCUMULATE_SUCCESS_CONTINUE = 1;      // 0 1
            const int ACCUMULATE_FAILURE_BREAK = 2;         // 1 0
            const int ACCUMULATE_FAILURE_CONTINUE = 3;      // 1 1

            groute::graphs::dev::CSRGraph m_csr_graph;
            groute::graphs::dev::CSCGraph m_csc_graph;
            uint32_t m_nnodes;
            uint32_t m_nedges;
            common::Model m_model;
            common::MsgPassing m_msg_passing;
            common::Scheduling m_scheduling;
            uint32_t *m_p_current_round;

            __forceinline__ __device__
            virtual TValue GetInitValue(index_t node) const = 0;

            __forceinline__ __device__
            virtual TBuffer GetInitBuffer(index_t node) const = 0;

            __forceinline__ __host__ __device__
            virtual TBuffer GetIdentityElement() const = 0;

            /**
             * Combine the value of node and the buffer, and return new buffer to push or be pulled by neighbor
             * @param node
             * @param p_value
             * @param p_buffer
             * @return
             */
            __forceinline__ __device__
            virtual utils::pair<TBuffer, bool> CombineValueBuffer(index_t node,
                                                                  TValue *p_value,
                                                                  TBuffer *p_buffer) = 0;

            __forceinline__ __device__
            virtual int AccumulateBuffer(index_t src,
                                         index_t dst,
                                         TBuffer *p_buffer,
                                         TBuffer buffer)
            {
                assert(false);
                return 0;
            }

            __forceinline__ __device__
            virtual int AccumulateBuffer(index_t src,
                                         index_t dst,
                                         TWeight weight,
                                         TBuffer *p_buffer,
                                         TBuffer buffer)
            {
                assert(false);
                return 0;
            }

            /**
             * Deterimine a node is active or not.
             * This method will be called for Topology-driven variants.
             * @param node node id
             * @param buffer the buffer of node
             * @return return true if node is active.
             */
            __forceinline__ __device__
            virtual bool IsActiveNode(index_t node, TBuffer buffer, TValue value) const
            {
                assert(false);
                return true;
            }

             __forceinline__ __device__
            virtual TValue sum_value(index_t node, TValue value, TBuffer buffer) const
            {
                assert(false);
                return 0;
            }

            __forceinline__ __device__
            virtual void PostComputation()
            {
                
            }

            __forceinline__ __device__
            virtual bool IsHighPriority(TBuffer current_priority, TBuffer buffer) const
            {
                return true;
            }

            __host__
            void SetVariant(common::Model model, common::MsgPassing msg_passing, common::Scheduling scheduling)
            {
                m_model = model;
                m_msg_passing = msg_passing;
                m_scheduling = scheduling;
            }

            __host__
            void SetVariant(common::AlgoVariant algo_variant)
            {
                m_model = algo_variant.m_model;
                m_msg_passing = algo_variant.m_msg_passing;
                m_scheduling = algo_variant.m_scheduling;
            }
        };
    }
}

#endif //HYBRID_API_H
