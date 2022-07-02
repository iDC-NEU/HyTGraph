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
#include "hybrid_bfs_common.h"


DEFINE_int32(source_node, 0, "The source node for the BFS traversal (clamped to [0, nnodes-1])");
DEFINE_bool(sparse, false, "use async/push/dd + fusion for high-diameter");
DECLARE_bool(non_atomic);
DECLARE_int32(top_ranks);
DECLARE_bool(print_ranks);
DECLARE_string(output);
DECLARE_bool(check);

namespace hybrid_bfs
{
    template<typename TValue, typename TBuffer, typename TWeight, typename...UnusedData>
    struct BFS : sepgraph::api::AppBase<TValue, TBuffer, TWeight>
    {
        using sepgraph::api::AppBase<TValue, TBuffer, TWeight>::AccumulateBuffer;
        index_t m_source_node;
        bool m_non_atomic;

        BFS(index_t source, bool non_atomic) : m_source_node(source), m_non_atomic(non_atomic)
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
            TBuffer buffer;

            if (node == m_source_node)
            {
                buffer = 0;
            }
            else
            {
                buffer = UINT32_MAX;
            }

            return buffer;
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
                buffer += 1;
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
            //return buffer != IDENTITY_ELEMENT;
            return value > buffer;        
        }
    };
}

bool HybridBFS()
{
    LOG("HybridBFS\n");
    typedef sepgraph::engine::Engine<level_t, level_t, groute::graphs::NoWeight, hybrid_bfs::BFS, index_t, bool> HybridEngine;
    HybridEngine engine(sepgraph::policy::AlgoType::TRAVERSAL_SCHEME); //host_graph ready
    
    engine.LoadGraph();
    index_t source_node = min(max((index_t) 0, (index_t) FLAGS_source_node), engine.GetGraphDatum().nnodes - 1);//source_node not more than nnodes - 1
    
    sepgraph::common::EngineOptions engine_opt;
    auto regression = BFSHost(engine.CSRGraph(), source_node);

    
    engine.SetOptions(engine_opt);
    engine.InitGraph(source_node, FLAGS_non_atomic);
    engine.Start();
    engine.PrintInfo();

    utils::JsonWriter &writer = utils::JsonWriter::getInst();

    writer.write("non_atomic", FLAGS_non_atomic ? "YES" : "NO");

    const auto &levels = engine.GatherValue();
    bool success = true;

    if (FLAGS_check)
    {
        int errors = BFSCheckErrors(levels, regression);

        success = errors == 0;
        printf("total errors: %d\n", errors);
    }
    else
    {
        printf("Warning: Result not checked\n");
    }

    if (FLAGS_output.length() > 0)
    {
        BFSOutput(FLAGS_output.data(), levels);
    }
    return true;
}
