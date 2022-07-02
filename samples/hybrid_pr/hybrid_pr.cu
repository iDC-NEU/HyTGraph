// ----------------------------------------------------------------
// SEP-Graph: Finding Shortest Execution Paths for Graph Processing under a Hybrid Framework on GPU
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE
// in the root directory of this source distribution.
// ----------------------------------------------------------------
#include <framework/framework.cuh>
#include <framework/hybrid_policy.h>
#include <framework/clion_cuda.cuh>
#include <framework/variants/api.cuh>
#include "hybrid_pr_common.h"
#include <functional>
#include <map>

// Priority
DEFINE_double(cut_threshold, 0, "Cut threshold for index calculation");
DEFINE_bool(sparse, false, "disable load-balancing for sparse graph");
DECLARE_double(error);
DECLARE_int32(top_ranks);
DECLARE_bool(print_ranks);
DECLARE_string(output);

namespace hybrid_pr
{
    template<typename TValue, typename TBuffer, typename TWeight, typename...UnusedData>
    struct PageRank : sepgraph::api::AppBase<TValue, TBuffer, TWeight>
    {

        /*
         * For get rid of compiler bug: It's strange that if base class has virtual function, we must add a member for subclass.
         *
         * Error: Internal Compiler Error (codegen): "there was an error in verifying the lgenfe output!"
         */
        double m_error;
        using sepgraph::api::AppBase<TValue, TBuffer, TWeight>::AccumulateBuffer;

        PageRank(double error) : m_error(error)
        {
	      
        }

        __forceinline__ __device__

        TValue GetInitValue(index_t node) const override
        {
            return 0.0f;
        }

        __forceinline__ __device__

        TBuffer GetInitBuffer(index_t node) const override
        {
            return 1 - ALPHA;
        }

        __forceinline__ __host__
        __device__
                TBuffer

        GetIdentityElement() const override
        {
            return 0.0f;
        }

        __forceinline__ __device__

        utils::pair<TBuffer, bool> CombineValueBuffer(index_t node,
                                                      TValue *p_value,
                                                      TBuffer *p_buffer) override
        {
            TBuffer buffer = atomicExch(p_buffer, IDENTITY_ELEMENT);
            bool schedule = false;
            if (buffer > 0.01)
            {
                schedule = true;
                *p_value += buffer;

                int out_degree = this->m_csr_graph.end_edge(node) -
                                     this->m_csr_graph.begin_edge(node);
                buffer = ALPHA * buffer / out_degree;

            }

            return utils::pair<TBuffer, bool>(buffer, schedule);
        }

        __forceinline__ __device__

        int AccumulateBuffer(index_t src,
                             index_t dst,
                             TBuffer *p_buffer,
                             TBuffer buffer) override
        {
            atomicAdd(p_buffer, buffer);
            return 0;
        }

        __forceinline__ __device__

        bool IsActiveNode(index_t node, TBuffer buffer, TValue value) const override
        {
            return buffer > m_error;
        }
        
        __forceinline__ __device__

        TValue sum_value(index_t node, TValue value, TBuffer buffer) const override
        {
            return buffer;
        }

        __forceinline__ __device__

        bool IsHighPriority(TBuffer current_priority, TBuffer buffer) const override
        {
            return current_priority <= buffer;
        }
    };
}

bool HybridPageRank()
{
    LOG("HybridPageRank\n");
    typedef sepgraph::engine::Engine<rank_t, rank_t, groute::graphs::NoWeight, hybrid_pr::PageRank, double> HybridEngine;
    HybridEngine engine(sepgraph::policy::AlgoType::ITERATIVE_SCHEME);
    sepgraph::engine::EngineOptions engine_opt;


    engine.SetOptions(engine_opt);
    engine.LoadGraph();
    engine.InitGraph(FLAGS_error);
    engine.Start();
    engine.PrintInfo();
    utils::JsonWriter &writer = utils::JsonWriter::getInst();

    writer.write("error_tolerance", (float) FLAGS_error);

    const std::vector<rank_t> &ranks = engine.GatherValue();
    const std::vector<rank_t> &residual = engine.GatherBuffer();
    rank_t max_residual = *std::max_element(residual.begin(), residual.end());

    double pr_sum = 0;

    for (rank_t rank:ranks)
    {
        pr_sum += rank;
    }

    //printf("Total rank : %f\n", pr_sum);
    //printf("max residual: %f\n", max_residual);

    bool success = true;
    if (FLAGS_check)
    {
        auto regression = PageRankHost(engine.CSRGraph());
        auto gathered_output = engine.GatherValue();
        int errors = PageRankCheckErrors(gathered_output, regression);

        success = errors == 0;
        printf("total errors: %d\n", errors);
    }
    else
    {
        printf("Warning: Result not checked\n");
    }

    if (FLAGS_output.length() > 0)
    {
        PageRankOutput(FLAGS_output.data(), ranks);
    }

    return success;
}