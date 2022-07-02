// ----------------------------------------------------------------
// SEP-Graph: Finding Shortest Execution Paths for Graph Processing under a Hybrid Framework on GPU
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE
// in the root directory of this source distribution.
// ----------------------------------------------------------------
#include <cub/grid/grid_barrier.cuh>
#include <groute/common.h>
#include <groute/device/cta_scheduler_hybrid.cuh>
#include <groute/device/worklist_stack.cuh>
#include <utils/to_json.h>
#include <framework/variants/api.cuh>
#include <framework/framework.cuh>
#include <framework/hybrid_policy.h>
#include "hybrid_bc_common.h"

DEFINE_int32(source_node, 0, "The source node for the BFS traversal (clamped to [0, nnodes-1])");
DEFINE_bool(sparse, false, "use async/push/dd + fusion for high-diameter");
DECLARE_bool(check);
typedef sepgraph::dev::WorklistStack<index_t> WLStackDeviceObject;

namespace hybrid_bc {
    template<typename TValue, typename TBuffer, typename TWeight, typename...UnusedData>
    struct Stage1 : sepgraph::api::AppBase<TValue, TBuffer, TWeight> {
        using sepgraph::api::AppBase<TValue, TBuffer, TWeight>::AccumulateBuffer;
        index_t m_source_node;
        groute::graphs::dev::GraphDatum<float> m_node_sigmas;
        index_t *m_p_search_depth;
        WLStackDeviceObject m_wl_stack;

        Stage1(index_t source_node,
               const groute::graphs::dev::GraphDatum<float> node_sigmas,
               index_t *p_search_depth,
               WLStackDeviceObject wl_stack) :
                m_source_node(source_node),
                m_node_sigmas(node_sigmas),
                m_p_search_depth(p_search_depth),
                m_wl_stack(wl_stack) {
        }

        __forceinline__ __device__
        TValue GetInitValue(index_t node) const override {
            return static_cast<TValue> (UINT32_MAX);
        }

        __forceinline__ __device__
        TBuffer GetInitBuffer(index_t node) const {
            TBuffer buffer;

            if (node == m_source_node) {
                buffer = 0;
            } else {
                buffer = UINT32_MAX;
            }

            return buffer;
        }

        __forceinline__ __host__ __device__
        TBuffer GetIdentityElement() const {
            return UINT32_MAX;
        }


        __forceinline__ __device__
        utils::pair<TBuffer, bool> CombineValueBuffer(index_t node,
                                                      TValue *p_value,
                                                      TBuffer *p_buffer) override {
            TBuffer buffer = *p_buffer;//atomicExch(p_buffer, UINT32_MAX);
            bool schedule;

            if (this->m_msg_passing == sepgraph::common::MsgPassing::PUSH) {
                schedule = false;

                if (*p_value > buffer) {
                    *p_value = buffer;
                    buffer += 1;
                    schedule = true;
                    m_wl_stack.append(node);
                }
            } else {
                schedule = true;

                if (*p_value > buffer) {
                    *p_value = buffer;
                    buffer += 1;
                    // If pull mode is used, we only pull from neighbor one time.
                    schedule = false;
                    m_wl_stack.append(node);
                } else if (*p_buffer != IDENTITY_ELEMENT) {
                    // same story as above
                    schedule = false;
                }
            }

            return utils::pair<TBuffer, bool>(buffer, schedule);
        }

        __forceinline__ __device__
        int AccumulateBuffer(index_t src,
                             index_t dst,
                             TBuffer *p_buffer,
                             TBuffer buffer) override {
            TBuffer prev = atomicMin(p_buffer, buffer);

            if (prev == IDENTITY_ELEMENT) {
                atomicAdd(m_node_sigmas.get_item_ptr(dst), m_node_sigmas[src]);
                atomicMax(m_p_search_depth, buffer);
            } else {
                if (*p_buffer == buffer) {
                    atomicAdd(m_node_sigmas.get_item_ptr(dst), m_node_sigmas[src]);
                }
            }

            if (buffer < prev) {
                return this->ACCUMULATE_SUCCESS_CONTINUE;
            } else {
                return this->ACCUMULATE_FAILURE_CONTINUE;
            }
        }

        __forceinline__ __device__
        bool IsActiveNode(index_t node, TBuffer buffer) const override {
            return buffer != UINT32_MAX;
        }

        __forceinline__ __device__
        void PostComputation() override {
            m_wl_stack.push();
        }
    };


    template<typename Graph,
            typename WLStack,
            typename SourcePath,
            typename Sigmas>
    __global__ void StageTwoDDFused(Graph graph,
                                    WLStack wl_stack,
                                    SourcePath node_source_path,
                                    Sigmas node_sigmas,
                                    Sigmas node_bc_values,
                                    uint32_t max_depth,
                                    cub::GridBarrier barrier) {
        uint32_t tid = TID_1D;
        uint32_t nthreads = TOTAL_THREADS_1D;
        uint32_t curr_depth = max_depth;


        while (curr_depth > 0) {
            assert(curr_depth >= 0);
            uint32_t begin_pos = wl_stack.begin_pos(curr_depth);
            uint32_t end_pos = wl_stack.end_pos(curr_depth);

            for (uint32_t idx = tid + begin_pos; idx < end_pos; idx += nthreads) {
                index_t node = wl_stack.read(idx);
                index_t src_depth = node_source_path[node];

                for (index_t edge = graph.begin_edge(node), end_edge = graph.end_edge(node);
                     edge < end_edge; edge++) {
                    index_t dest = graph.edge_dest(edge);

                    if (node_source_path[dest] == src_depth + 1) {
                        float delta_to = 1.0f * node_sigmas[node] / node_sigmas[dest] * (1.0f + node_bc_values[dest]);

                        atomicAdd(node_bc_values.get_item_ptr(node), delta_to);
                    }
                }
            }
            barrier.Sync();
            curr_depth--;
        }
    }

    template<typename Graph,
            typename WLStack,
            typename SourcePath,
            typename Sigmas>
    __global__ void StageTwoDDCTAFused(Graph graph,
                                       WLStack wl_stack,
                                       SourcePath node_source_path,
                                       Sigmas node_sigmas,
                                       Sigmas node_bc_values,
                                       uint32_t max_depth,
                                       cub::GridBarrier barrier) {
        uint32_t tid = TID_1D;
        uint32_t nthreads = TOTAL_THREADS_1D;
        uint32_t curr_depth = max_depth;

        while (curr_depth > 0) {
            uint32_t begin_pos = wl_stack.begin_pos(curr_depth);
            uint32_t end_pos = wl_stack.end_pos(curr_depth);
            uint32_t work_size = end_pos - begin_pos;
            uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;

            for (uint32_t i = 0 + tid; i < work_size_rup; i += nthreads) {
                groute::dev::np_local<index_t> np_local = {0, 0};

                if (i < work_size) {
                    index_t node = wl_stack.read(begin_pos + i);

                    np_local.start = graph.begin_edge(node);
                    np_local.size = graph.end_edge(node) - np_local.start;
                    np_local.meta_data = node;
                }

                groute::dev::CTAWorkSchedulerNew<index_t, groute::dev::LB_FINE_GRAINED>::template
                schedule(np_local,
                         [&graph, &node_source_path, &node_sigmas, &node_bc_values](index_t edge, index_t node) {
                             index_t src_depth = node_source_path[node];
                             index_t dest = graph.edge_dest(edge);

                             if (node_source_path[dest] == src_depth + 1) {
                                 float delta_to =
                                         1.0f * node_sigmas[node] / node_sigmas[dest] * (1.0f + node_bc_values[dest]);

                                 atomicAdd(node_bc_values.get_item_ptr(node), delta_to);
                             }
                             return true;
                         });
            }
            barrier.Sync();
            curr_depth--;
        }
    }
}

void Stage2(const groute::graphs::dev::CSRGraph &dev_csr_graph,
            const groute::graphs::single::NodeOutputDatum<level_t> &dev_levels_datum,
            utils::SharedArray<float> &dev_node_sigmas,
            utils::SharedArray<float> &dev_node_bc_values,
            const sepgraph::WorklistStack<index_t> &dev_wl_stack,
            uint32_t depth,
            const groute::Stream &stream) {
    cub::GridBarrierLifetime barrier;
    dim3 grid_dims, block_dims;
    int max_sm_occupancy;
    int dev_id = 0;
    cudaDeviceProp props;

    GROUTE_CUDA_CHECK(cudaGetDeviceProperties(&props, dev_id));
    uint32_t *d_p_tmp;

    GROUTE_CUDA_CHECK(cudaMalloc(&d_p_tmp, sizeof(uint32_t)));

    if (FLAGS_sparse) {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &max_sm_occupancy,
                hybrid_bc::StageTwoDDFused<groute::graphs::dev::CSRGraph,
                        sepgraph::dev::WorklistStack<index_t>,
                        groute::graphs::dev::GraphDatum<float>,
                        groute::graphs::dev::GraphDatum<float>>,
                FLAGS_block_size,
                0);
    } else {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &max_sm_occupancy,
                hybrid_bc::StageTwoDDCTAFused<groute::graphs::dev::CSRGraph,
                        sepgraph::dev::WorklistStack<index_t>,
                        groute::graphs::dev::GraphDatum<float>,
                        groute::graphs::dev::GraphDatum<float>>,
                FLAGS_block_size,
                0);
    }


    grid_dims.x = max_sm_occupancy * props.multiProcessorCount;
    block_dims.x = FLAGS_block_size;

    barrier.Setup(grid_dims.x);

    Stopwatch sw_stage2(true);

    if (FLAGS_sparse) {
        hybrid_bc::StageTwoDDFused << < grid_dims, block_dims, 0, stream.cuda_stream >> > (dev_csr_graph,
                dev_wl_stack.DeviceObject(),
                dev_levels_datum.DeviceObject(),
                groute::graphs::dev::GraphDatum<float>(dev_node_sigmas.dev_ptr, dev_node_sigmas.buffer_size),
                groute::graphs::dev::GraphDatum<float>(dev_node_bc_values.dev_ptr, dev_node_bc_values.buffer_size),
                depth,
                barrier);
    } else {
        hybrid_bc::StageTwoDDCTAFused << < grid_dims, block_dims, 0, stream.cuda_stream >> > (dev_csr_graph,
                dev_wl_stack.DeviceObject(),
                dev_levels_datum.DeviceObject(),
                groute::graphs::dev::GraphDatum<float>(dev_node_sigmas.dev_ptr, dev_node_sigmas.buffer_size),
                groute::graphs::dev::GraphDatum<float>(dev_node_bc_values.dev_ptr, dev_node_bc_values.buffer_size),
                depth,
                barrier);
    }
    stream.Sync();

    sw_stage2.stop();

    printf("Time stage2: %f\n", sw_stage2.ms());
    utils::JsonWriter &writer = utils::JsonWriter::getInst();

    writer.write("time_stage2", (float) sw_stage2.ms());
    writer.write("time_total", (float) (writer.get_float("time_total") + sw_stage2.ms()));
    printf("Time total: %f\n", writer.get_float("time_total"));
}

bool HybridBC() {
    index_t source_node = (index_t) FLAGS_source_node;

    typedef sepgraph::engine::Engine<level_t, level_t, groute::graphs::NoWeight,
            hybrid_bc::Stage1,
            index_t,                                        // source node
            const groute::graphs::dev::GraphDatum<float>,   // node sigmas
            uint32_t *,                                     // search depth
            WLStackDeviceObject
    > Hybridengine;

    Hybridengine engine(sepgraph::policy::AlgoType::TRAVERSAL_SCHEME);

    engine.LoadGraph();

    index_t nnodes = engine.CSRGraph().nnodes;
    source_node = min(max((index_t) 0, source_node), engine.GetGraphDatum().nnodes - 1);

    utils::SharedArray<sigma_t> dev_node_sigmas(nnodes);
    utils::SharedArray<float> dev_node_bc_values(nnodes);
    utils::SharedValue<uint32_t> dev_search_depth;

    dev_node_sigmas.host_vec[source_node] = 1;
    dev_node_sigmas.H2D();

    sepgraph::common::EngineOptions engine_opt;

    if (FLAGS_sparse) {
        engine_opt.SetFused();
        engine_opt.ForceVariant(sepgraph::common::AlgoVariant::ASYNC_PUSH_DD);
        engine_opt.SetLoadBalancing(sepgraph::common::MsgPassing::PUSH, sepgraph::common::LoadBalancing::NONE);
    } else {
        engine_opt.SetLoadBalancing(sepgraph::common::MsgPassing::PUSH, sepgraph::common::LoadBalancing::FINE_GRAINED);
        engine_opt.SetLoadBalancing(sepgraph::common::MsgPassing::PULL, sepgraph::common::LoadBalancing::FINE_GRAINED);
    }
    engine.SetOptions(engine_opt);

    sepgraph::WorklistStack<index_t> wl_stack(nnodes);

    const groute::Stream &stream = engine.getStream();

    wl_stack.ResetAsync(stream);

    WLStackDeviceObject wl_stack_dev_obj = wl_stack.DeviceObject();

    engine.InitGraph(source_node,
                     groute::graphs::dev::GraphDatum<float>(dev_node_sigmas.dev_ptr, dev_node_sigmas.buffer_size),
                     dev_search_depth.dev_ptr,
                     wl_stack_dev_obj);

    engine.Start();
    engine.PrintInfo();

    uint32_t depth = dev_search_depth.get_val_D2H() - 1;

    // Believe or not, you must wrap these code in a separate function. Otherwise, it will causes cudaOccupancyMaxActiveBlocksPerMultiprocessor to calculate incorrect result
    // ***** which means DEADLOCK (for high-diameter dataset) ****
    Stage2(engine.CSRDeviceObject(),
           engine.GetGraphDatum().m_node_value_datum,
           dev_node_sigmas,
           dev_node_bc_values,
           wl_stack,
           depth,
           stream);

    dev_node_sigmas.D2H();
    dev_node_bc_values.D2H();

    std::vector<float> node_sigmas = dev_node_sigmas.host_vec;
    std::vector<float> node_bc_values = dev_node_bc_values.host_vec;

    for (index_t node = 0; node < engine.CSRGraph().nnodes; node++) {
        node_bc_values[node] /= 2;
    }

    bool success;

    if (FLAGS_check) {
        auto bc_sigma = BetweennessCentralityHost(engine.CSRGraph(), source_node);
        std::vector<centrality_t> &regression_bc_values = bc_sigma.first;
        std::vector<sigma_t> &regression_sigma_values = bc_sigma.second;

        uint32_t topK = std::min(10u, nnodes);

        for (index_t node = 0; node < topK; node++) {
            printf("%u %f %f\n", node, regression_bc_values[node], node_bc_values[node]);
        }

        printf("Checking sigmas...\n");
        success = BCCheckErrors(regression_sigma_values, node_sigmas) == 0;
        if (!success) {
            fprintf(stderr, "Error sigmas");
        }

        printf("Checking BC values...\n");
        success = success && BCCheckErrors(regression_bc_values, node_bc_values) == 0;
        if (!success) {
            fprintf(stderr, "Error BC  values");
        }
    } else {
        printf("Warning: Result not checked\n");
        success = true;
    }

    if (FLAGS_output.length() > 0) {
        BCOutput(FLAGS_output.c_str(), node_bc_values);
    }

    return success;
}