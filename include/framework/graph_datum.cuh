// ----------------------------------------------------------------
// SEP-Graph: Finding Shortest Execution Paths for Graph Processing under a Hybrid Framework on GPU
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE
// in the root directory of this source distribution.
// ----------------------------------------------------------------
#ifndef HYBRID_GRAPH_DATUM_H
#define HYBRID_GRAPH_DATUM_H

#include <gflags/gflags.h>
#include <framework/common.h>
#include <framework/hybrid_policy.h>
#include <groute/device/bitmap_impls.h>
#include <groute/graphs/csr_graph.h>
#include <groute/device/queue.cuh>
#include <utils/cuda_utils.h>
#include <vector>



#define PRIORITY_SAMPLE_SIZE 1000

DECLARE_int32(SEGMENT);
DECLARE_double(wl_alloc_factor);

namespace sepgraph {
    namespace graphs {
        template<typename TValue,
                typename TBuffer,
                typename TWeight>
        struct GraphDatum {
            // Graph metadata
            uint32_t nnodes, nedges;
	               
            index_t segment = FLAGS_SEGMENT;
            // Worklist
	        groute::Queue<index_t> m_wl_array_in_seg[512];
            groute::Queue<index_t> m_wl_array_in; // Work-list in
            groute::Queue<index_t> m_wl_array_out_high; // Work-list out High priority
            groute::Queue<index_t> m_wl_array_out_low; // Work-list out Low priority
            groute::Queue<index_t> m_wl_middle;
	    
	        //groute::Queue<index_t> m_wl_array_seg;
	        //std::vector< groute::Queue<index_t> > m_wl_array_total;
	        /*Code by AX range 39 to 41*/
	        std::vector<index_t> seg_active_num;
	        std::vector<index_t> seg_workload_num;
	        std::vector<TValue> seg_res_num;
            std::vector<index_t> seg_exc_list;
	    
            Bitmap m_wl_bitmap_in; // Work-list in
            Bitmap m_wl_bitmap_out_high; // Work-list out high
            Bitmap m_wl_bitmap_out_low; // Work-list out low
            Bitmap m_wl_bitmap_middle;

            utils::SharedValue<uint32_t> m_current_round;

            // In/Out-degree for every nodes
            utils::SharedArray<uint32_t> m_in_degree;
            utils::SharedArray<uint32_t> m_out_degree;

            // Total In/Out-degree
            utils::SharedValue<uint32_t> m_total_in_degree;
            utils::SharedValue<uint32_t> m_total_out_degree;
	        utils::SharedValue<uint32_t> m_seg_degree;
	        utils::SharedValue<TValue> m_seg_value;

            // Graph data
            groute::graphs::single::NodeOutputDatum<TValue> m_node_value_datum;
            groute::graphs::single::NodeOutputDatum<TBuffer> m_node_buffer_datum;
            groute::graphs::single::NodeOutputDatum<TBuffer> m_node_buffer_tmp_datum; // For sync algorithms
            groute::graphs::single::EdgeInputDatum<TWeight> m_csr_edge_weight_datum;
            groute::graphs::single::EdgeInputDatum<TWeight> m_csc_edge_weight_datum;

            //Subgraph data
            utils::SharedArray<uint32_t> activeNodesLabeling;
            utils::SharedArray<uint32_t> activeNodesDegree;
            utils::SharedArray<uint32_t> prefixLabeling;
            utils::SharedArray<uint32_t> prefixSumDegrees;
            uint32_t subgraphnodes,subgraphedges;
            uint32_t Compaction_num;
            // Running data
            utils::SharedValue<uint32_t> m_active_nodes;

            // Sampling
            utils::SharedArray<index_t> m_sampled_nodes;
            utils::SharedArray<TBuffer> m_sampled_values;
            bool m_weighted;
            bool m_on_pinned_memory;
            index_t priority_detal;

            GraphDatum(const groute::graphs::host::CSRGraph &csr_graph,
                       //const groute::graphs::host::CSCGraph &csc_graph,
		               uint64_t seg_max_edge,
                       std::vector<index_t> nnodes_num,
                                             bool OnPinnedMemory=true) : nnodes(csr_graph.nnodes),
                                                                          nedges(csr_graph.nedges),
                                                                          m_in_degree(nullptr, 0),
                                                                          m_out_degree(nullptr, 0),
                                                                          activeNodesDegree(nullptr, 0),
                                                                          activeNodesLabeling(nullptr, 0),
                                                                          prefixLabeling(nullptr, 0),
                                                                          prefixSumDegrees(nullptr, 0),
                                                                          m_sampled_nodes(nullptr, 0),
                                                                          m_sampled_values(nullptr, 0),
                                                                          m_on_pinned_memory(OnPinnedMemory){
                m_node_value_datum.Allocate(csr_graph);
                m_node_buffer_datum.Allocate(csr_graph);
                m_node_buffer_tmp_datum.Allocate(csr_graph);

                // Weighted graph
                if (typeid(TWeight) != typeid(groute::graphs::NoWeight)) {
		            m_csr_edge_weight_datum.Allocate_node(csr_graph,seg_max_edge);
                    m_weighted = true;
                }
                else{
		             m_weighted = false;
		        }   

                uint32_t capacity = nnodes * FLAGS_wl_alloc_factor;

		        for(index_t i = 0; i < segment; i++){
		              m_wl_array_in_seg[i] = std::move(groute::Queue<index_t>(nnodes_num[i]));
		        }
                m_wl_array_in_seg[segment] = std::move(groute::Queue<index_t>(nnodes)); //for zero task combine
                m_wl_array_in_seg[segment + 1] = std::move(groute::Queue<index_t>(nnodes)); //for compaction task combine

		        seg_active_num = std::move(std::vector<index_t>(segment));
		        seg_workload_num = std::move(std::vector<index_t>(segment));
		        seg_res_num = std::move(std::vector<TValue>(segment));
		        seg_exc_list = std::move(std::vector<index_t>(segment));

                GROUTE_CUDA_CHECK(cudaMalloc(&activeNodesLabeling.dev_ptr, nnodes * sizeof(uint32_t)));
                GROUTE_CUDA_CHECK(cudaMalloc(&activeNodesDegree.dev_ptr, nnodes * sizeof(uint32_t)));
                GROUTE_CUDA_CHECK(cudaMalloc(&prefixLabeling.dev_ptr, nnodes * sizeof(uint32_t)));
                GROUTE_CUDA_CHECK(cudaMalloc(&prefixSumDegrees.dev_ptr, (nnodes + 1) * sizeof(uint32_t)));

                m_sampled_nodes = std::move(utils::SharedArray<index_t>(PRIORITY_SAMPLE_SIZE));
                m_sampled_values = std::move(utils::SharedArray<TBuffer>(PRIORITY_SAMPLE_SIZE));
            }

            GraphDatum(GraphDatum &&other) = delete;

            GraphDatum &operator=(const GraphDatum &other) = delete;

            GraphDatum &operator=(GraphDatum &&other) = delete;

            const groute::graphs::dev::GraphDatum<TValue> &GetValueDeviceObject() const {
                return m_node_value_datum.DeviceObject();
            }

            const groute::graphs::dev::GraphDatum<TBuffer> &GetBufferDeviceObject() const {
                return m_node_buffer_datum.DeviceObject();
            }

            const groute::graphs::dev::GraphDatum<TBuffer> &GetBufferTmpDeviceObject() const {
                return m_node_buffer_tmp_datum.DeviceObject();
            }

            const groute::graphs::dev::GraphDatum<TWeight> &GetEdgeWeightDeviceObject() const {
                return m_csr_edge_weight_datum.DeviceObject();
            }

            const groute::graphs::dev::GraphDatum<TWeight> &GetCSCEdgeWeightDeviceObject() const {
                return m_csc_edge_weight_datum.DeviceObject();
            }

            const groute::dev::WorkSourceRange<index_t> GetWorkSourceRangeDeviceObject(index_t seg_snode, index_t seg_nnode) {
                return groute::dev::WorkSourceRange<index_t>(seg_snode, seg_nnode);
            }
	        const groute::dev::WorkSourceRange<index_t> GetWorkSourceRangeDeviceObject() {
                return groute::dev::WorkSourceRange<index_t>(0, nnodes);
            }
            
            const std::vector<TValue> &GatherValue() {
                m_node_value_datum.Gather();
                return m_node_value_datum.GetHostData();
            }

            const std::vector<TBuffer> &GatherBuffer() {
                m_node_buffer_datum.Gather();
                return m_node_buffer_datum.GetHostData();
            }
        };
    }
}

#endif //HYBRID_GRAPH_DATUM_H
