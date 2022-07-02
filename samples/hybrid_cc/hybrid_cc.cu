// ----------------------------------------------------------------
// SEP-Graph: Finding Shortest Execution Paths for Graph Processing under a Hybrid Framework on GPU
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE
// in the root directory of this source distribution.
// ----------------------------------------------------------------
#include <functional>
#include <map>
//#define ARRAY_BITMAP
#include <framework/framework.cuh>
#include <framework/hybrid_policy.h>
#include <framework/clion_cuda.cuh>
#include <framework/variants/api.cuh>
#include <framework/common.h>
#include "hybrid_cc_common.h"

DEFINE_bool(sparse, false, "use async/push/dd + fusion for high-diameter");
DECLARE_bool(non_atomic);
DECLARE_int32(top_ranks);
DECLARE_bool(print_ranks);
DECLARE_string(output);
DECLARE_bool(check);

namespace hybrid_cc
{
    template<typename TValue, typename TBuffer, typename TWeight, typename...UnusedData>
    struct CC : sepgraph::api::AppBase<TValue, TBuffer, TWeight>
    {
        using sepgraph::api::AppBase<TValue, TBuffer, TWeight>::AccumulateBuffer;
        bool m_non_atomic;

        CC(bool non_atomic) :m_non_atomic(non_atomic)
        {

        }

        __forceinline__ __device__

        TValue GetInitValue(index_t node) const override
        {
            return static_cast<TValue> (IDENTITY_ELEMENT);
        }

        __forceinline__ __device__

        TBuffer GetInitBuffer(index_t node) const override
        {
            return TBuffer(node);
        }

        __forceinline__ __host__
        __device__
        TBuffer

        GetIdentityElement() const override
        {
            return IDENTITY_ELEMENT;
        }

        __forceinline__ __device__

        utils::pair<TBuffer, bool> CombineValueBuffer(index_t node,
                                                      TValue *p_value,
                                                      TBuffer *p_buffer) override
        {
            TBuffer buffer = *p_buffer;
            bool schedule;

                schedule = false;

            if (*p_value > buffer)
            {
                *p_value = buffer;
                schedule = true;
            }
            return utils::pair<TBuffer, bool>(buffer, schedule);
        }

        __forceinline__ __device__
        int AccumulateBuffer(index_t src,
                             index_t dst,
                             TBuffer *p_buffer,
                             TBuffer buffer) override
        {            
            atomicMin(p_buffer, buffer);    
            return 0;
        }
        
        __forceinline__ __device__
        TValue sum_value(index_t node, TValue value, TBuffer buffer) const override
        {
            if(value > buffer * 2)
                return TValue(2);

            return TValue(1);
        }

        __forceinline__ __device__

        bool IsActiveNode(index_t node, TBuffer buffer,TValue value) const override
        {
            return buffer < value;
        }
    };
}

bool HybridCC()
{
    LOG("HybridCC\n");
    typedef sepgraph::engine::Engine<level_t, level_t, groute::graphs::NoWeight, hybrid_cc::CC,index_t> HybridEngine;
    HybridEngine engine(sepgraph::policy::AlgoType::TRAVERSAL_SCHEME); //host_graph ready
    
    engine.LoadGraph();
    
    sepgraph::common::EngineOptions engine_opt;
    
    index_t source_node = 0;
    engine.SetOptions(engine_opt);
    engine.InitGraph(source_node);
    engine.Start();
    engine.PrintInfo();

    utils::JsonWriter &writer = utils::JsonWriter::getInst();
    
    return true;
}
