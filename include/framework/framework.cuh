// ----------------------------------------------------------------
// SEP-Graph: Finding Shortest Execution Paths for Graph Processing under a Hybrid Framework on GPU
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE
// in the root directory of this source distribution.
// ----------------------------------------------------------------
#ifndef HYBRID_FRAMEWORK_H
#define HYBRID_FRAMEWORK_H

#include <functional>
#include <map>
#include <math.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thread>
#include <thrust/sort.h>
#include <cub/cub.cuh>
#include <framework/common.h>
#include <framework/variants/api.cuh>
#include <framework/graph_datum.cuh>
#include <framework/variants/common.cuh>
#include <framework/variants/driver.cuh>
#include <framework/hybrid_policy.h>
#include <framework/algo_variants.cuh>
#include <utils/cuda_utils.h>
#include <utils/graphs/traversal.h>
#include <utils/to_json.h>
#include <groute/device/work_source.cuh>
#include "clion_cuda.cuh"

DECLARE_int32(residence);
DECLARE_int32(priority_a);
DECLARE_int32(hybrid);
DECLARE_int32(SEGMENT);
DECLARE_int32(n_stream);
DECLARE_int32(max_iteration);
DECLARE_string(out_wl);
DECLARE_string(lb_push);
DECLARE_string(lb_pull);
DECLARE_double(alpha);
DECLARE_bool(undirected);
DECLARE_bool(wl_sort);
DECLARE_bool(wl_unique);
DECLARE_double(edge_factor);

namespace sepgraph {
    namespace engine {
        using common::Priority;
        using common::LoadBalancing;
        using common::Scheduling;
        using common::Model;
        using common::MsgPassing;
        using common::AlgoVariant;
        using policy::AlgoType;
        using policy::PolicyDecisionMaker;
        using utils::JsonWriter;

        struct Algo {
            static const char *Name() {
                return "Hybrid Graph Engine";
            }
        };  



      template<typename TValue, typename TBuffer, typename TWeight, template<typename, typename, typename, typename ...> class TAppImpl, typename... UnusedData>
      class 	Engine {
        private:
        typedef TAppImpl<TValue, TBuffer, TWeight, UnusedData...> AppImplDeviceObject;
        typedef graphs::GraphDatum<TValue, TBuffer, TWeight> GraphDatum;

        cudaDeviceProp m_dev_props;
	    
            // Graph data
            std::unique_ptr<utils::traversal::Context<Algo>> m_groute_context;
            std::unique_ptr<groute::Stream> m_stream;
            std::unique_ptr<groute::graphs::single::CSRGraphAllocator> m_csr_dev_graph_allocator;
            std::unique_ptr<groute::graphs::single::CSCGraphAllocator> m_csc_dev_graph_allocator;

            std::unique_ptr<AppImplDeviceObject> m_app_inst;
            groute::Stream stream[64];
            // App instance
            std::unique_ptr<GraphDatum> m_graph_datum;

            policy::TRunningInfo m_running_info;
            TBuffer current_priority;  // put it into running info
            PolicyDecisionMaker m_policy_decision_maker;
            EngineOptions m_engine_options;

            //partition_information
            unsigned int partitions_csc;
            unsigned int* partition_offset_csc;
            unsigned int max_partition_size_csc;

            unsigned int partitions_csr;
            unsigned int* partition_offset_csr;
            unsigned int max_partition_size_csr;


            public:
            Engine(AlgoType algo_type) :
            m_running_info(algo_type),
            m_policy_decision_maker(m_running_info) {
                int dev_id = 0;

                GROUTE_CUDA_CHECK(cudaGetDeviceProperties(&m_dev_props, dev_id));
                m_groute_context = std::unique_ptr<utils::traversal::Context<Algo>>
                (new utils::traversal::Context<Algo>(1));

		        //create stream /*CODE by ax range 118 to 121*/
                for(int i = 0; i < FLAGS_n_stream; i++)
                {
                    stream[i] = m_groute_context->CreateStream(dev_id);
                }
		
        m_stream = std::unique_ptr<groute::Stream>(new groute::Stream(dev_id));
    }

    void SetOptions(EngineOptions &engine_options) {
        m_engine_options = engine_options;
    }

    index_t GetNodeNum(){
      return m_groute_context->host_graph.nnodes;
  }

  void LoadGraph() {
      Stopwatch sw_load(true);

      groute::graphs::host::CSRGraph &csr_graph = m_groute_context->host_graph;
      LOG("Converting CSR to CSC...\n");
      index_t m_nsegs = FLAGS_SEGMENT;

      uint64_t seg_sedge_csr, seg_eedge_csr;
      index_t seg_snode,seg_enode, seg_nnodes;
      uint64_t seg_nedges_csr;
      uint64_t seg_nedges = round_up(csr_graph.nedges, m_nsegs);

	  uint64_t seg_nedges_csr_max = 0;  //dev memory		    
	  uint64_t edge_num = 0;		    
	  index_t node_id = 0;
	  uint64_t out_degree;
	  std::vector<index_t> nnodes_num;
      uint32_t maxnode_num;
	  seg_snode = node_id;
	  m_groute_context->seg_snode[0] = seg_snode;
	  for(index_t seg_idx = 0; node_id < csr_graph.nnodes ; seg_idx++){
          m_groute_context->seg_snode[seg_idx] = node_id;
          while(edge_num < seg_nedges){
             out_degree = csr_graph.end_edge(node_id) - csr_graph.begin_edge(node_id);
             edge_num = edge_num + out_degree;
             if(node_id < csr_graph.nnodes){
                    node_id++;
             }
             else{
                break; 
             }
           }

           if(node_id == csr_graph.nnodes){
                seg_enode = node_id; 
            }
           else{
                seg_enode = node_id;	    
            }

            seg_nnodes = seg_enode - seg_snode; 
            m_running_info.nnodes_seg[seg_idx] = seg_nnodes;
            nnodes_num.push_back(seg_nnodes);

            m_groute_context->seg_enode[seg_idx] = seg_enode;
            seg_sedge_csr = csr_graph.row_start[seg_snode];                            // start edge		    
            seg_eedge_csr = csr_graph.row_start[seg_enode];                            // end edge
            seg_nedges_csr = seg_eedge_csr - seg_sedge_csr;	

            m_running_info.total_workload_seg[seg_idx] = seg_nedges_csr;
            seg_nedges_csr_max = max(seg_nedges_csr_max,seg_nedges_csr);

            m_groute_context->seg_sedge_csr[seg_idx] = seg_sedge_csr; 
            m_groute_context->seg_nedge_csr[seg_idx] = seg_nedges_csr;

            edge_num = 0;
            seg_snode = node_id;		     
        }
                m_groute_context->segment_ct = FLAGS_SEGMENT;
                m_groute_context->SetDevice(0);

                m_csr_dev_graph_allocator = std::unique_ptr<groute::graphs::single::CSRGraphAllocator>(
                    new groute::graphs::single::CSRGraphAllocator(csr_graph,seg_nedges_csr_max));

                m_graph_datum = std::unique_ptr<GraphDatum>(new GraphDatum(csr_graph,seg_nedges_csr_max,nnodes_num));

                sw_load.stop();

                m_running_info.time_load_graph = sw_load.ms();

                LOG("Load graph time: %f ms (excluded)\n", sw_load.ms());

                m_running_info.nnodes = m_groute_context->nvtxs;
                m_running_info.nedges = m_groute_context->nedges;
                m_running_info.total_workload = m_groute_context->nedges * FLAGS_edge_factor;
                current_priority = m_engine_options.GetPriorityThreshold();
            }

            /*
             * Init Graph Value and buffer fields
             */
             void InitGraph(UnusedData &...data) {
                Stopwatch sw_init(true);

                m_app_inst = std::unique_ptr<AppImplDeviceObject>(new AppImplDeviceObject(data...));

                groute::Stream &stream_s = *m_stream;
                GraphDatum &graph_datum = *m_graph_datum;

                const auto &dev_csr_graph = m_csr_dev_graph_allocator->DeviceObject();
                const auto &work_source = groute::dev::WorkSourceRange<index_t>(0, graph_datum.nnodes);
                dim3 grid_dims, block_dims;

                m_app_inst->m_csr_graph = dev_csr_graph;
                m_app_inst->m_nnodes = graph_datum.nnodes;
                m_app_inst->m_nedges = graph_datum.nedges;
                m_app_inst->m_p_current_round = graph_datum.m_current_round.dev_ptr;

                // Launch kernel to init value/buffer fields
                KernelSizing(grid_dims, block_dims, work_source.get_size());

                auto &app_inst = *m_app_inst;
                kernel::InitGraph
                << < grid_dims, block_dims, 0, stream_s.cuda_stream >> > (app_inst,
                    work_source,
                    graph_datum.GetValueDeviceObject(),
                    graph_datum.GetBufferDeviceObject(),
                    graph_datum.GetBufferTmpDeviceObject());
                stream_s.Sync();

                index_t seg_snode,seg_enode;
                index_t stream_id;

                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT; seg_idx++){
                         stream_id = seg_idx % FLAGS_n_stream;
                         seg_snode = m_groute_context->seg_snode[seg_idx];                                    // start node
                         seg_enode = m_groute_context->seg_enode[seg_idx];  
                         RebuildArrayWorklist(app_inst,
                            graph_datum,
                        stream[stream_id],seg_snode,seg_enode - seg_snode,seg_idx);
                    }
                

                for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                    stream[stream_idx].Sync();
                }

               m_groute_context->host_graph.PrintHistogram(graph_datum.m_in_degree.dev_ptr,graph_datum.m_out_degree.dev_ptr);
               m_running_info.time_init_graph = sw_init.ms();

               sw_init.stop();

               LOG("InitGraph: %f ms (excluded)\n", sw_init.ms());	
           }

           void SaveToJson() {
            JsonWriter &writer = JsonWriter::getInst();

            writer.write("time_input_active_node", m_running_info.time_overhead_input_active_node);
            writer.write("time_output_active_node", m_running_info.time_overhead_output_active_node);
            writer.write("time_input_workload", m_running_info.time_overhead_input_workload);
            writer.write("time_output_workload", m_running_info.time_overhead_output_workload);
            writer.write("time_queue2bitmap", m_running_info.time_overhead_queue2bitmap);
            writer.write("time_bitmap2queue", m_running_info.time_overhead_bitmap2queue);
            writer.write("time_rebuild_worklist", m_running_info.time_overhead_rebuild_worklist);
            writer.write("time_priority_sample", m_running_info.time_overhead_sample);
            writer.write("time_sort_worklist", m_running_info.time_overhead_wl_sort);
            writer.write("time_unique_worklist", m_running_info.time_overhead_wl_unique);
            writer.write("time_kernel", m_running_info.time_kernel);
            writer.write("time_total", m_running_info.time_total);
            writer.write("time_per_round", m_running_info.time_total / m_running_info.current_round);
            writer.write("num_iteration", (int) m_running_info.current_round);

            if (m_engine_options.IsForceVariant()) {
                writer.write("force_variant", m_engine_options.GetAlgoVariant().ToString());
            }

            if (m_engine_options.IsForceLoadBalancing(MsgPassing::PUSH)) {
                writer.write("force_push_load_balancing",
                 LBToString(m_engine_options.GetLoadBalancing(MsgPassing::PUSH)));
            }

            if (m_engine_options.IsForceLoadBalancing(MsgPassing::PULL)) {
                writer.write("force_pull_load_balancing",
                 LBToString(m_engine_options.GetLoadBalancing(MsgPassing::PULL)));
            }

            if (m_engine_options.GetPriorityType() == Priority::NONE) {
                writer.write("priority_type", "none");
                } else if (m_engine_options.GetPriorityType() == Priority::LOW_HIGH) {
                    writer.write("priority_type", "low_high");
                    writer.write("priority_delta", m_engine_options.GetPriorityThreshold());
                    } else if (m_engine_options.GetPriorityType() == Priority::SAMPLING) {
                        writer.write("priority_type", "sampling");
                        writer.write("cut_threshold", m_engine_options.GetCutThreshold());
                    }

                    writer.write("fused_kernel", m_engine_options.IsFused() ? "YES" : "NO");
                    writer.write("max_iteration_reached",
                     m_running_info.current_round == 1000 ? "YES" : "NO");
                //writer.write("date", get_now());
                writer.write("device", m_dev_props.name);
                writer.write("dataset", FLAGS_graphfile);
                writer.write("nnodes", (int) m_graph_datum->nnodes);
                writer.write("nedges", (int) m_graph_datum->nedges);
                writer.write("algo_type", m_running_info.m_algo_type == AlgoType::TRAVERSAL_SCHEME ? "TRAVERSAL_SCHEME"
                   : "ITERATIVE_SCHEME");
            }

            void PrintInfo() {
                LOG("--------------Overhead--------------\n");
                LOG("Rebuild worklist: %f\n", m_running_info.time_overhead_rebuild_worklist);
                LOG("Priority sample: %f\n", m_running_info.time_overhead_sample);
                LOG("hybrid Worlist: %f\n", m_running_info.time_overhead_hybrid);
                LOG("Unique Worklist: %f\n", m_running_info.time_overhead_wl_unique);
                LOG("--------------Time statistics---------\n");
                LOG("Kernel time: %f\n", m_running_info.time_kernel);
                LOG("Total time: %f\n", m_running_info.time_total);
                LOG("Total rounds: %d\n", m_running_info.current_round);
                LOG("Time/round: %f\n", m_running_info.time_total / m_running_info.current_round);
                LOG("filter_num: %d\n", m_running_info.explicit_num);
                LOG("zerocopy_num: %d\n", m_running_info.zerocopy_num);
                LOG("compaction_num: %d\n", m_running_info.compaction_num);


                LOG("--------------Engine info-------------\n");
                if (m_engine_options.IsForceVariant()) {
                    LOG("Force variant: %s\n", m_engine_options.GetAlgoVariant().ToString().data());
                }

                if (m_engine_options.IsForceLoadBalancing(MsgPassing::PUSH)) {
                    LOG("Force Push Load balancing: %s\n",
                        LBToString(m_engine_options.GetLoadBalancing(MsgPassing::PUSH)).data());
                }

                if (m_engine_options.IsForceLoadBalancing(MsgPassing::PULL)) {
                    LOG("Force Pull Load balancing: %s\n",
                        LBToString(m_engine_options.GetLoadBalancing(MsgPassing::PULL)).data());
                }

                if (m_engine_options.GetPriorityType() == Priority::NONE) {
                    LOG("Priority type: NONE\n");
                    } else if (m_engine_options.GetPriorityType() == Priority::LOW_HIGH) {
                        LOG("Priority type: LOW_HIGH\n");
                        LOG("Priority delta: %f\n", m_engine_options.GetPriorityThreshold());
                        } else if (m_engine_options.GetPriorityType() == Priority::SAMPLING) {
                            LOG("Priority type: Sampling\n");
                            LOG("Cut threshold: %f\n", m_engine_options.GetCutThreshold());
                        }

                        LOG("Fused kernel: %s\n", m_engine_options.IsFused() ? "YES" : "NO");
                        LOG("Max iteration reached: %s\n", m_running_info.current_round == 1000 ? "YES" : "NO");


                        LOG("-------------Misc-------------------\n");
                //LOG("Date: %s\n", get_now().data());
                LOG("Device: %s\n", m_dev_props.name);
                LOG("Dataset: %s\n", FLAGS_graphfile.data());
                LOG("Algo type: %s\n",
                    m_running_info.m_algo_type == AlgoType::TRAVERSAL_SCHEME ? "TRAVERSAL_SCHEME" : "ITERATIVE_SCHEME");
            }

            void Start(index_t priority_detal = 0) {
                
                GraphDatum &graph_datum = *m_graph_datum;
                graph_datum.priority_detal = priority_detal;

                AlgoVariant next_policy[FLAGS_SEGMENT];
                for(index_t i = 0; i < FLAGS_SEGMENT; i++){
                    next_policy[i] = m_policy_decision_maker.GetInitPolicy();
                }
                bool convergence = false;
                Stopwatch sw_total(true);
                LoadOptions();

                while (!convergence) {
                  PreComputationBW();	    
                  CombineTask(next_policy);
                  ExecutePolicyBW(next_policy); 	    
                  for(index_t i = 0; i < FLAGS_SEGMENT; i++){
                    next_policy[i] = m_policy_decision_maker.GetNextPolicy(i,graph_datum.Compaction_num);
                  }

                  int convergence_check = 0;
                  for(index_t seg_id = 0; seg_id < FLAGS_SEGMENT; seg_id++){
                       if(m_running_info.input_active_count_seg[seg_id] == 0){
                           convergence_check++;
                       }
                   }
                  if(convergence_check == FLAGS_SEGMENT){
                        convergence = true;
                  }
                  if (m_running_info.current_round == 1000 ) {//FLAGS_max_iteration
                        convergence = true;
                        LOG("Max iterations reached\n");
                  }
                    
                }

                sw_total.stop();

                m_running_info.time_total = sw_total.ms();

            }

            const groute::graphs::host::CSRGraph &CSRGraph() const {
                return m_groute_context->host_graph;
            }

            const GraphDatum &GetGraphDatum() const {
                return *m_graph_datum;
            }

            const std::vector<TValue> &GatherValue() {
                return m_graph_datum->GatherValue();
            }

            const std::vector<TValue> &GatherBuffer() {
                return m_graph_datum->GatherBuffer();
            }

            groute::graphs::dev::CSRGraph CSRDeviceObject() const {
                return m_csr_dev_graph_allocator->DeviceObject();
            }

            const groute::Stream &getStream() const {
                return *m_stream;
            }

            private:
            void LoadOptions() {
                if (!m_engine_options.IsForceLoadBalancing(MsgPassing::PUSH)) {
                    if (FLAGS_lb_push.size() == 0) {
                        if (m_groute_context->host_graph.avg_degree() >= 0) { //all FINE_GRAINED
                            m_engine_options.SetLoadBalancing(MsgPassing::PUSH, LoadBalancing::FINE_GRAINED);
                        }
                        } else {
                            if (FLAGS_lb_push == "none") {
                                m_engine_options.SetLoadBalancing(MsgPassing::PUSH, LoadBalancing::NONE);
                                } else if (FLAGS_lb_push == "coarse") {
                                    m_engine_options.SetLoadBalancing(MsgPassing::PUSH, LoadBalancing::COARSE_GRAINED);
                                    } else if (FLAGS_lb_push == "fine") {
                                        m_engine_options.SetLoadBalancing(MsgPassing::PUSH, LoadBalancing::FINE_GRAINED);
                                        } else if (FLAGS_lb_push == "hybrid") {
                                            m_engine_options.SetLoadBalancing(MsgPassing::PUSH, LoadBalancing::HYBRID);
                                            } else {
                                                fprintf(stderr, "unknown push load-balancing policy");
                                                exit(1);
                                            }
                                        }
                                    }

                if (!m_engine_options.IsForceLoadBalancing(MsgPassing::PULL)) {
                    if (FLAGS_lb_pull.size() == 0) {
                        if (m_groute_context->host_graph.avg_degree() >= 5) {
                            m_engine_options.SetLoadBalancing(MsgPassing::PULL, LoadBalancing::FINE_GRAINED);
                        }
                        } else {
                            if (FLAGS_lb_pull == "none") {
                                m_engine_options.SetLoadBalancing(MsgPassing::PULL, LoadBalancing::NONE);
                                } else if (FLAGS_lb_pull == "coarse") {
                                    m_engine_options.SetLoadBalancing(MsgPassing::PULL, LoadBalancing::COARSE_GRAINED);
                                    } else if (FLAGS_lb_pull == "fine") {
                                        m_engine_options.SetLoadBalancing(MsgPassing::PULL, LoadBalancing::FINE_GRAINED);
                                        } else if (FLAGS_lb_pull == "hybrid") {
                                            m_engine_options.SetLoadBalancing(MsgPassing::PULL, LoadBalancing::HYBRID);
                                            } else {
                                                fprintf(stderr, "unknown pull load-balancing policy");
                                                exit(1);
                                            }
                                        }
                                    }

                                    if (FLAGS_alpha == 0) {
                                        fprintf(stderr, "Warning: alpha = 0, A general method AsyncPushDD is used\n");
                                        m_engine_options.ForceVariant(AlgoVariant::ASYNC_PUSH_DD);
                                    }
                                }


            void PreComputationBW() {// Reorganizing nothing, just reset the round and record the workload. 
                const int dev_id = 0;
                const groute::Stream &stream = m_groute_context->CreateStream(dev_id);
                GraphDatum &graph_datum = *m_graph_datum;
                m_running_info.current_round++;
                graph_datum.m_current_round.set_val_H2DAsync(m_running_info.current_round, stream.cuda_stream);                

                stream.Sync();
            }

            void ExecutePolicyBW(AlgoVariant *algo_variant) {
                auto &app_inst = *m_app_inst;
                auto &csr_graph_host = m_csr_dev_graph_allocator->HostObject();
                GraphDatum &graph_datum = *m_graph_datum;
                Stopwatch sw_execution(true);
                Stopwatch sw_round(true);
                bool zcflag = true;
		        m_csr_dev_graph_allocator->AllocateDevMirror_Edge_Zero();

		        uint64_t seg_sedge_csr,seg_nedges_csr;
		        index_t seg_snode,seg_enode;
		        std::pair<index_t, TValue> seg_res_idx;
		        std::vector<std::pair<index_t, TValue>> seg_res_rank;   
                index_t seg_idx_new;
                Stopwatch sw_priority(true);
                if(FLAGS_priority_a == 1){
                    for(index_t seg_idx = 0; seg_idx < m_groute_context->segment_ct; seg_idx++){
                            seg_idx_new = m_groute_context->segment_id_ct[seg_idx];
                            seg_res_idx.first = seg_idx;    
                            seg_res_idx.second = graph_datum.seg_res_num[seg_idx_new];
                            seg_res_rank.push_back(seg_res_idx);
                    }
                    std::sort(seg_res_rank.begin(), seg_res_rank.end(), [](std::pair<index_t, TValue> v1, std::pair<index_t, TValue> v2){
                            return v1.second > v2.second;
                        }); 
                }

                index_t priority_seg = m_groute_context->segment_ct;
                
                index_t seg_exc = 0;
                for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                     m_graph_datum->seg_exc_list[stream_idx] = -1;
                }
                sw_priority.stop();
                m_running_info.time_overhead_sample += sw_priority.ms();

                index_t stream_id;
                for(index_t seg_idx = 0; seg_idx < priority_seg ; seg_idx++){
                    if(FLAGS_priority_a == 1){
                        seg_idx_new = seg_res_rank[seg_idx].first;
                    }
		            else{
                        seg_idx_new = seg_idx;
                    }
                    uint32_t seg_idx_exp = m_groute_context->segment_id_ct[seg_idx_new];
    		        seg_snode = m_groute_context->seg_snode[seg_idx_exp];                                    // start node
    		        seg_enode = m_groute_context->seg_enode[seg_idx_exp];                                    // end node
    		        seg_sedge_csr = m_groute_context->seg_sedge_csr[seg_idx_exp];                            // start edge
    		        seg_nedges_csr = m_groute_context->seg_nedge_csr[seg_idx_exp]; 
    		        
                    //printf("seg_idx_new:%d seg_snode:%d seg_enode:%d seg_sedge_csr:%lu seg_nedges_csr:%lu \n",seg_idx_new,seg_snode,seg_enode,seg_sedge_csr,seg_nedges_csr);
    		        stream_id = seg_idx_new % FLAGS_n_stream;
                    const auto &csr_graph = m_csr_dev_graph_allocator->DeviceObject();
                    if(algo_variant[seg_idx_new] == AlgoVariant::Exp_Filter){
                        //printf("exp\n");
                        m_running_info.explicit_num++;
                        seg_exc++;
                        m_graph_datum->seg_exc_list[stream_id] = seg_idx_new; 
                        m_csr_dev_graph_allocator->AllocateDevMirror_Edge_Explicit_Step(seg_nedges_csr,seg_sedge_csr,stream[stream_id],stream_id);
                        m_csr_dev_graph_allocator->SwitchExp(stream_id);
                        if(m_graph_datum->m_weighted == true){
                            m_graph_datum->m_csr_edge_weight_datum.AllocateDevMirror_edge_explicit_step(csr_graph_host,seg_nedges_csr,seg_sedge_csr,stream[stream_id],stream_id);   
                            m_graph_datum->m_csr_edge_weight_datum.SwitchExp(stream_id);
                        }
                        zcflag = false;
                        RunSyncPushDDB(app_inst,seg_snode,seg_enode,seg_sedge_csr,seg_idx_exp,zcflag,
                           csr_graph,
                           graph_datum,
                           m_engine_options,
                           stream[stream_id]);  
                    }
                    else if(algo_variant[seg_idx_new] == AlgoVariant::Zero_Copy){
                        //printf("zero\n");
                        m_running_info.zerocopy_num++;
                        m_csr_dev_graph_allocator->SwitchZC();
                        m_graph_datum->m_csr_edge_weight_datum.SwitchZC();
                        zcflag = true;
                        seg_idx_new = FLAGS_SEGMENT;
                        RunSyncPushDDB(app_inst,seg_snode,seg_enode,seg_sedge_csr,seg_idx_new,zcflag,
                           csr_graph,
                           graph_datum,
                           m_engine_options,
                           stream[stream_id]); 
                    }
                    else{
                        //printf("Compaction\n");
                        m_running_info.compaction_num++;
                        Compaction();
                        m_csr_dev_graph_allocator->AllocateDevMirror_Edge_Compaction(graph_datum.subgraphedges,stream[stream_id]);
                        m_csr_dev_graph_allocator->SwitchCom();
                        if(m_graph_datum->m_weighted == true){
                            m_graph_datum->m_csr_edge_weight_datum.AllocateDevMirror_edge_compaction(csr_graph_host,graph_datum.subgraphedges,stream[stream_id]);
                            m_graph_datum->m_csr_edge_weight_datum.SwitchCom();
                        }
                        zcflag = false;
                        seg_idx_new = FLAGS_SEGMENT + 1;
                        RunSyncPushDDB_COM(app_inst,
                           graph_datum.subgraphnodes,
                           seg_idx_new,
                           csr_graph,
                           graph_datum,
                           m_engine_options,
                           stream[stream_id]); 
                    }
		          //LOG("PUSHDD RUN in round(%d)\t|batch(%d)\t|engine(%s)\t|range(%d,%d)\n",m_running_info.current_round,seg_idx_new,zcflag?"ZC":"Exp",seg_snode,seg_enode);
               }

               for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                     stream[stream_idx].Sync();
               }
               sw_round.stop();
               if(seg_exc >= FLAGS_n_stream && FLAGS_residence == 1){
                
               for(index_t seg_idx = 0; seg_idx < FLAGS_n_stream; seg_idx++){
                    if(m_graph_datum->seg_exc_list[seg_idx] != -1){
                            index_t seg_exc = m_graph_datum->seg_exc_list[seg_idx];
                            stream_id = seg_exc % FLAGS_n_stream;
                            uint32_t seg_idx_exp = m_groute_context->segment_id_ct[seg_idx];
                            seg_snode = m_groute_context->seg_snode[seg_idx_exp];                                    // start node
                            seg_enode = m_groute_context->seg_enode[seg_idx_exp];  
                            RebuildArrayWorklist(app_inst,
                                graph_datum,
                                stream[stream_id],seg_snode,seg_enode - seg_snode,seg_idx_exp);

                    }
                    else{
                        continue;
                    }
                }

                for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                            stream[stream_idx].Sync();
                }

                for(index_t seg_idx = 0; seg_idx < FLAGS_n_stream ; seg_idx++){
                    if(m_graph_datum->seg_exc_list[seg_idx] != -1){
                    index_t seg_exc = m_graph_datum->seg_exc_list[seg_idx];
                    seg_idx_new = seg_exc;
                    uint32_t seg_idx_exp = m_groute_context->segment_id_ct[seg_idx_new];
                    stream_id = seg_exc % FLAGS_n_stream;
                    seg_snode = m_groute_context->seg_snode[seg_idx_exp];                                    // start node
                    seg_enode = m_groute_context->seg_enode[seg_idx_exp];                                    // end node
                    seg_sedge_csr = m_groute_context->seg_sedge_csr[seg_idx_exp];                            // start edge
                    seg_nedges_csr = m_groute_context->seg_nedge_csr[seg_idx_exp]; 
                        
                    m_csr_dev_graph_allocator->SwitchExp(stream_id);
                    if(m_graph_datum->m_weighted == true){ 
                        m_graph_datum->m_csr_edge_weight_datum.SwitchExp(stream_id);
                    }
                    zcflag = false;
                    auto csr_graph = m_csr_dev_graph_allocator->DeviceObject();

                    //LOG("PUSHDD RUN in round(%d)\t|batch(%d)\t|engine(%s)\t|range(%d,%d)\n",m_running_info.current_round,seg_idx_new,zcflag?"ZC":"Exp",seg_snode,seg_enode);
                    RunSyncPushDDB(app_inst,seg_snode,seg_enode,seg_sedge_csr,seg_idx_exp,zcflag,
                       csr_graph,
                       graph_datum,
                       m_engine_options,
                       stream[stream_id]); 
                    }
                    else{
                        continue;
                    }            
                }
                for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                    stream[stream_idx].Sync();
                    m_graph_datum->seg_exc_list[stream_idx] = -1;
                }
            }
          
          m_running_info.time_round = sw_round.ms();	
          PostComputationBW();
          sw_execution.stop();

          m_running_info.time_kernel += sw_execution.ms();
}


            void PostComputationBW() {
                int dev_id = 0;
                const groute::Stream &stream_seg = m_groute_context->CreateStream(dev_id);
                GraphDatum &graph_datum = *m_graph_datum;
                AppImplDeviceObject &app_inst = *m_app_inst;
                m_running_info.current_round = m_graph_datum->m_current_round.get_val_D2H();

                Stopwatch sw_unique(true);

                index_t seg_snode,seg_enode;
                index_t stream_id;

                Stopwatch sw_rebuild(true);
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT; seg_idx++){
                    stream_id = seg_idx % FLAGS_n_stream;
			        seg_snode = m_groute_context->seg_snode[seg_idx];                                    // start node
			        seg_enode = m_groute_context->seg_enode[seg_idx];  
			        RebuildArrayWorklist(app_inst,
                        graph_datum,
                        stream[stream_id],seg_snode,seg_enode - seg_snode,seg_idx);
                }

                for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                    stream[stream_idx].Sync();
                }
                sw_rebuild.stop();
                m_running_info.time_overhead_rebuild_worklist += sw_rebuild.ms();
                
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT ; seg_idx++){
                    stream_id = seg_idx % FLAGS_n_stream;		      

		            index_t active_count = graph_datum.m_wl_array_in_seg[seg_idx].GetCount(stream[stream_id]);	    
		            graph_datum.seg_active_num[seg_idx] = active_count;

		            m_running_info.input_active_count_seg[seg_idx] = active_count;

		            uint32_t work_size = active_count;
		            dim3 grid_dims, block_dims;

                    if(FLAGS_hybrid == 2){
                        Stopwatch sw_hybrid(true);
		                  graph_datum.m_seg_degree.set_val_H2DAsync(0, stream[stream_id].cuda_stream);
		                  KernelSizing(grid_dims, block_dims, work_size);

		                  kernel::SumOutDegreeQueue << < grid_dims, block_dims, 0, stream[stream_id].cuda_stream >> >
                          (groute::dev::WorkSourceArray<index_t>(
                            graph_datum.m_wl_array_in_seg[seg_idx].GetDeviceDataPtr(),
                                work_size),
                            graph_datum.m_out_degree.dev_ptr,
                            graph_datum.m_seg_degree.dev_ptr);
                            stream[stream_id].Sync();

                        graph_datum.seg_workload_num[seg_idx] = graph_datum.m_seg_degree.get_val_D2H();
                        m_running_info.input_workload_seg[seg_idx] = graph_datum.seg_workload_num[seg_idx];
                        sw_hybrid.stop();
                        m_running_info.time_overhead_hybrid += sw_hybrid.ms();
                    }

                    /*if(FLAGS_priority_a == 1){
                            Stopwatch sw_priority(true);
                            graph_datum.m_seg_value.set_val_H2DAsync(0, stream[stream_id].cuda_stream);
                            KernelSizing(grid_dims, block_dims, work_size);
                            kernel::SumResQueue << < grid_dims, block_dims, 0, stream[stream_id].cuda_stream >> >
                                (app_inst,
                                    groute::dev::WorkSourceArray<index_t>(
                                    graph_datum.m_wl_array_in_seg[seg_idx].GetDeviceDataPtr(),
                                    work_size),
                                graph_datum.GetValueDeviceObject(),
                                graph_datum.GetBufferDeviceObject(),
                                graph_datum.m_seg_value.dev_ptr);
                                
                                stream[stream_id].Sync();

                                graph_datum.seg_res_num[seg_idx] = graph_datum.m_seg_value.get_val_D2H();
                            sw_priority.stop();
                            m_running_info.time_overhead_sample += sw_priority.ms();
                    }*/

                }
                sw_unique.stop();
                m_running_info.time_overhead_wl_unique += sw_unique.ms();

          }

           void CombineTask(AlgoVariant *algo_variant) {
                int dev_id = 0;
                const groute::Stream &stream_seg = m_groute_context->CreateStream(dev_id);
                GraphDatum &graph_datum = *m_graph_datum;
                AppImplDeviceObject &app_inst = *m_app_inst;
                graph_datum.m_wl_array_in_seg[FLAGS_SEGMENT].ResetAsync(stream_seg.cuda_stream);
                graph_datum.m_wl_array_in_seg[FLAGS_SEGMENT + 1].ResetAsync(stream_seg.cuda_stream);
                stream_seg.Sync();
                Stopwatch sw_unique(true);
                index_t seg_snode,seg_enode;
                index_t stream_id;
                index_t task = 1;// zero:0 exp_filter:1 exp_compaction:2
                bool zc = false;
                bool compaction = false;
                Stopwatch sw_rebuild(true);
                index_t seg_idx_ct = 0;
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT; seg_idx++){  
                    stream_id = seg_idx % FLAGS_n_stream;
                    seg_snode = m_groute_context->seg_snode[seg_idx];
                    if(algo_variant[seg_idx] == AlgoVariant::Zero_Copy){
                        task = 0;
                        zc = true;
                        //printf("zero\n");
                        while(algo_variant[seg_idx + 1] == AlgoVariant::Zero_Copy && seg_idx < FLAGS_SEGMENT - 1){
                            seg_idx++;
                        }
                    }
                    if(algo_variant[seg_idx] == AlgoVariant::Exp_Compaction){
                        task = 2;
                        compaction = true;
                        //printf("Compaction\n");
                        while(algo_variant[seg_idx + 1] == AlgoVariant::Exp_Compaction && seg_idx < FLAGS_SEGMENT - 1){
                            seg_idx++;
                        }
                    }
                    seg_enode = m_groute_context->seg_enode[seg_idx];
                    //printf("seg_idx:%d seg_snode:%d seg_enode:%d seg_sedge_csr:%lu seg_nedges_csr:%lu \n",seg_idx,seg_snode,seg_enode,m_groute_context->seg_sedge_csr_ct[seg_idx_ct],m_groute_context->seg_nedge_csr_ct[seg_idx_ct]); 
                    if(task == 0){
                        task = 1;
                        RebuildArrayWorklist_zero(app_inst,
                            graph_datum,
                            stream[stream_id],seg_snode,seg_enode - seg_snode,FLAGS_SEGMENT);
                    }
                    else if(task == 1)
                    {
                        algo_variant[seg_idx_ct] = AlgoVariant::Exp_Filter;
                        m_groute_context->segment_id_ct[seg_idx_ct++] = seg_idx;
                        RebuildArrayWorklist(app_inst,
                            graph_datum,
                            stream[stream_id],seg_snode,seg_enode - seg_snode,seg_idx);

                    }
                    else if(task == 2){
                        task = 1;
                        RebuildArrayWorklist_compaction(app_inst,
                            graph_datum,
                            stream[stream_id],seg_snode,seg_enode - seg_snode,FLAGS_SEGMENT + 1);
                    }
                }
                if(zc){
                    m_groute_context->segment_id_ct[seg_idx_ct] = seg_idx_ct;
                    algo_variant[seg_idx_ct++] = AlgoVariant::Zero_Copy;
                }
                if(compaction){
                    m_groute_context->segment_id_ct[seg_idx_ct] = seg_idx_ct;
                    algo_variant[seg_idx_ct++] = AlgoVariant::Exp_Compaction;
                }

                for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                    stream[stream_idx].Sync();
                }
                m_groute_context->segment_ct = seg_idx_ct;
                sw_rebuild.stop();
                m_running_info.time_overhead_rebuild_worklist += sw_rebuild.ms();
                for(index_t seg_idx = 0; seg_idx < seg_idx_ct ; seg_idx++){
                    uint32_t seg_idx_new = m_groute_context->segment_id_ct[seg_idx];
                    stream_id = seg_idx % FLAGS_n_stream;            
                    index_t active_count = graph_datum.m_wl_array_in_seg[seg_idx_new].GetCount(stream[stream_id]);  
                    uint32_t work_size = active_count;
                    dim3 grid_dims, block_dims;

                    if(FLAGS_priority_a == 1){
                            Stopwatch sw_priority(true);
                            if(algo_variant[seg_idx_new] == AlgoVariant::Zero_Copy){
                                graph_datum.seg_res_num[seg_idx_new] = 1;
                                continue;
                            }
                            if(algo_variant[seg_idx_new] == AlgoVariant::Exp_Compaction){
                                graph_datum.seg_res_num[seg_idx_new] = 0;
                                continue;
                            }
                            graph_datum.m_seg_value.set_val_H2DAsync(0, stream[stream_id].cuda_stream);
                            KernelSizing(grid_dims, block_dims, work_size);
                            kernel::SumResQueue << < grid_dims, block_dims, 0, stream[stream_id].cuda_stream >> >
                                (app_inst,
                                    groute::dev::WorkSourceArray<index_t>(
                                    graph_datum.m_wl_array_in_seg[seg_idx_new].GetDeviceDataPtr(),
                                    work_size),
                                graph_datum.GetValueDeviceObject(),
                                graph_datum.GetBufferDeviceObject(),
                                graph_datum.m_seg_value.dev_ptr);
                                
                            stream[stream_id].Sync();
                            graph_datum.seg_res_num[seg_idx_new] = graph_datum.m_seg_value.get_val_D2H();
                            sw_priority.stop();
                            m_running_info.time_overhead_sample += sw_priority.ms();
                    }

                }
                graph_datum.Compaction_num = 0;
                sw_unique.stop();
                m_running_info.time_overhead_wl_unique += sw_unique.ms();

          }
            void com_test(     
                            uint32_t numActiveNodes,
                            uint32_t *activeNodes,
                            uint32_t *activeNodesPointer,
                            uint64_t *nodePointer, 
                            uint32_t *activeEdgeList,
                            uint32_t *edgeList)
            {

                uint32_t thisNode;
                uint32_t thisDegree;
                uint32_t fromHere;
                uint32_t fromThere;
                GraphDatum &graph_datum = *m_graph_datum;
                for(uint32_t i=0; i< graph_datum.subgraphnodes; i++)
                {
                    thisNode = activeNodes[i];
                    thisDegree = activeNodesPointer[i + 1] - activeNodesPointer[i];
                    fromHere = activeNodesPointer[i];
                    fromThere = nodePointer[thisNode];
                    for(uint32_t j=0; j<thisDegree; j++)
                    {
                        activeEdgeList[fromHere+j] = edgeList[fromThere+j];
                    }
                }
                
            }

            void dynamic(uint32_t tId,
                            uint32_t numThreads,    
                            uint32_t numActiveNodes,
                            uint32_t *activeNodes,
                            uint32_t *activeNodesPointer,
                            uint64_t *nodePointer, 
                            uint32_t *activeEdgeList,
                            uint32_t *edgeList)
            {

                uint32_t chunkSize = numActiveNodes / numThreads + 1;
                uint32_t left, right;
                
                left = tId * chunkSize;
                right = min(left+chunkSize, numActiveNodes);    
                uint32_t thisNode;
                uint32_t thisDegree;
                uint32_t fromHere;
                uint32_t fromThere;

                for(uint32_t i=left; i<right; i++)
                {
                    thisNode = activeNodes[i];
                    thisDegree = activeNodesPointer[i + 1] - activeNodesPointer[i];
                    fromHere = activeNodesPointer[i];
                    fromThere = nodePointer[thisNode];
                    for(uint32_t j=0; j<thisDegree; j++)
                    {
                        activeEdgeList[fromHere+j] = edgeList[fromThere+j];
                    }
                }
                
            }

            void dynamic_weight(uint32_t tId,
                            uint32_t numThreads,    
                            uint32_t numActiveNodes,
                            uint32_t *activeNodes,
                            uint32_t *activeNodesPointer,
                            uint64_t *nodePointer, 
                            uint32_t *activeEdgeList,
                            uint32_t *edgeList,
                            uint32_t *activeEdgeListWeight,
                            uint32_t *edgeListWeight)
            {

                uint32_t chunkSize = numActiveNodes / numThreads + 1;
                uint32_t left, right;

                left = tId * chunkSize;
                right = min(left+chunkSize, numActiveNodes);    
                
                uint32_t thisNode;
                uint32_t thisDegree;
                uint32_t fromHere;
                uint32_t fromThere;

                for(uint32_t i=left; i<right; i++)
                {
                    thisNode = activeNodes[i];
                    thisDegree = activeNodesPointer[i + 1] - activeNodesPointer[i];
                    fromHere = activeNodesPointer[i];
                    fromThere = nodePointer[thisNode];
                    for(uint32_t j=0; j<thisDegree; j++)
                    {
                        activeEdgeList[fromHere+j] = edgeList[fromThere+j];
                        activeEdgeListWeight[fromHere+j] = edgeListWeight[fromThere+j];
                    }
                }
                
            }
            void check(){
                GraphDatum &graph_datum = *m_graph_datum;
                auto &csr_graph_host = m_csr_dev_graph_allocator->HostObject();
                uint32_t thisNode;
                uint32_t thisDegree;
                uint32_t fromHere;
                uint32_t fromThere;
                uint32_t realDegree;
                uint32_t falseegde = 0;
                for(uint32_t i = 0;i < graph_datum.subgraphnodes; i++){

                    thisNode = csr_graph_host.subgraph_activenode[i];
                    thisDegree = csr_graph_host.subgraph_rowstart[i + 1] - csr_graph_host.subgraph_rowstart[i];
                    fromHere = csr_graph_host.subgraph_rowstart[i];
                    fromThere = csr_graph_host.row_start[thisNode];
                    realDegree = csr_graph_host.row_start[thisNode + 1] - fromThere;
                    for(uint32_t j = 0; j < thisDegree; j++){
                        if(csr_graph_host.subgraph_edgedst[fromHere + j] != csr_graph_host.edge_dst[fromThere + j] || csr_graph_host.subgraph_edgeweight[fromHere+j] != csr_graph_host.edge_weights[fromThere+j]){
                            //printf("Compaction edge:%d,real edge:%d\n",csr_graph_host.subgraph_edgedst[fromHere + j],csr_graph_host.edge_dst[fromThere + j]);
                            falseegde++;
                        }
                    }
                }
                //printf("falseegde:%d total_edge:%d\n",falseegde,graph_datum.subgraphedges);
            }
          void Compaction() {
                int dev_id = 0;
                const groute::Stream &stream_seg = m_groute_context->CreateStream(dev_id);
                GraphDatum &graph_datum = *m_graph_datum;
                AppImplDeviceObject &app_inst = *m_app_inst;
                auto csr_graph = m_csr_dev_graph_allocator->DeviceObject();
                auto &csr_graph_host = m_csr_dev_graph_allocator->HostObject();
                thrust::device_ptr<uint32_t> ptr_labeling(graph_datum.activeNodesLabeling.dev_ptr);
                thrust::device_ptr<uint32_t> ptr_labeling_prefixsum(graph_datum.prefixLabeling.dev_ptr);

                graph_datum.subgraphnodes = thrust::reduce(ptr_labeling, ptr_labeling + graph_datum.nnodes);

                thrust::exclusive_scan(ptr_labeling, ptr_labeling + graph_datum.nnodes, ptr_labeling_prefixsum);

                kernel::makeQueue<<<graph_datum.nnodes/512+1, 512>>>(csr_graph.subgraph_activenode, graph_datum.activeNodesLabeling.dev_ptr, graph_datum.prefixLabeling.dev_ptr, graph_datum.nnodes);

                GROUTE_CUDA_CHECK(cudaMemcpy(csr_graph_host.subgraph_activenode, csr_graph.subgraph_activenode, graph_datum.subgraphnodes*sizeof(uint32_t), cudaMemcpyDeviceToHost));

                thrust::device_ptr<uint32_t> ptr_degrees(graph_datum.activeNodesDegree.dev_ptr);
                thrust::device_ptr<uint32_t> ptr_degrees_prefixsum(graph_datum.prefixSumDegrees.dev_ptr);

                thrust::exclusive_scan(ptr_degrees, ptr_degrees + graph_datum.nnodes, ptr_degrees_prefixsum);

                kernel::makeActiveNodesPointer<<<graph_datum.nnodes/512+1, 512>>>(csr_graph.subgraph_rowstart, graph_datum.activeNodesLabeling.dev_ptr, graph_datum.prefixLabeling.dev_ptr, graph_datum.prefixSumDegrees.dev_ptr, graph_datum.nnodes);
                
                GROUTE_CUDA_CHECK(cudaMemcpy(csr_graph_host.subgraph_rowstart, csr_graph.subgraph_rowstart, graph_datum.subgraphnodes*sizeof(uint32_t), cudaMemcpyDeviceToHost));

                uint32_t numActiveEdges = 0;
                uint32_t endid = csr_graph_host.subgraph_activenode[graph_datum.subgraphnodes-1];
                uint32_t outDegree = csr_graph_host.end_edge(endid) - csr_graph_host.begin_edge(endid);
                if(graph_datum.subgraphnodes > 0)
                    numActiveEdges = csr_graph_host.subgraph_rowstart[graph_datum.subgraphnodes-1] + outDegree; 
                
                graph_datum.subgraphedges = numActiveEdges;
                uint32_t last = numActiveEdges;

                GROUTE_CUDA_CHECK(cudaMemcpy(csr_graph.subgraph_rowstart + graph_datum.subgraphnodes, &last, sizeof(uint32_t), cudaMemcpyHostToDevice));
    
                GROUTE_CUDA_CHECK(cudaMemcpy(csr_graph_host.subgraph_rowstart, csr_graph.subgraph_rowstart, (graph_datum.subgraphnodes + 1)*sizeof(uint32_t), cudaMemcpyDeviceToHost));

                uint32_t numThreads = 32;

                if(graph_datum.subgraphnodes < 5000)
                    numThreads = 1;
                std::thread runThreads[numThreads];
                if(m_graph_datum->m_weighted == false){
                    for(uint32_t t=0; t<numThreads; t++)
                    {   
                        uint32_t tid = t;
                        auto f = [&](uint32_t tid){dynamic(
                                                tid,
                                                numThreads,
                                                graph_datum.subgraphnodes,
                                                csr_graph_host.subgraph_activenode,
                                                csr_graph_host.subgraph_rowstart,
                                                csr_graph_host.row_start, 
                                                csr_graph_host.subgraph_edgedst,
                                                csr_graph_host.edge_dst);};
                        
                        runThreads[tid] = std::thread(f,tid);

                    }
                        
                    for(unsigned int t=0; t<numThreads; t++)
                        runThreads[t].join();
               }

               else{
                    for(uint32_t t=0; t<numThreads; t++)
                    {
                        uint32_t tid = t;
                        auto f = [&](uint32_t tid){dynamic_weight(
                                                tid,
                                                numThreads,
                                                graph_datum.subgraphnodes,
                                                csr_graph_host.subgraph_activenode, 
                                                csr_graph_host.subgraph_rowstart,
                                                csr_graph_host.row_start, 
                                                csr_graph_host.subgraph_edgedst,
                                                csr_graph_host.edge_dst,
                                                csr_graph_host.subgraph_edgeweight,
                                                csr_graph_host.edge_weights);};
                        
                        runThreads[tid] = std::thread(f,tid);
                    }
                        
                    for(unsigned int t=0; t<numThreads; t++)
                        runThreads[t].join();
               }
               // com_test(graph_datum.subgraphnodes,
               //                                   csr_graph_host.subgraph_activenode,
               //                                   csr_graph_host.subgraph_rowstart,
               //                                   csr_graph_host.row_start, 
               //                                   csr_graph_host.subgraph_edgedst,
               //                                   csr_graph_host.edge_dst);
          }

      };


  }
}

#endif //HYBRID_FRAMEWORK_H
