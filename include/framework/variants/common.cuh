// ----------------------------------------------------------------
// SEP-Graph: Finding Shortest Execution Paths for Graph Processing under a Hybrid Framework on GPU
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE
// in the root directory of this source distribution.
// ----------------------------------------------------------------
#ifndef HYBRID_KERNEL_COMMON_H
#define HYBRID_KERNEL_COMMON_H

namespace sepgraph
{
    namespace kernel
    {
        namespace common
        {
            // For array or range worklist
            template<typename TAppInst,
                    typename WorkSource,
                    typename WorkTarget,
                    template<typename> class GraphDatum,
                    typename TValue,
                    typename TBuffer>
            __global__
            void CombineValueBuffer(TAppInst app_inst,
                                    WorkSource work_source,
                                    WorkTarget work_target,
                                    GraphDatum<TValue> node_value_datum,
                                    GraphDatum<TBuffer> node_in_buffer_datum,
                                    GraphDatum<TBuffer> node_out_buffer_datum)
            {
                uint32_t tid = TID_1D;
                uint32_t nthreads = TOTAL_THREADS_1D;
                uint32_t work_size = work_source.get_size();
		 
		
                assert(node_value_datum.size == node_in_buffer_datum.size);
                assert(node_value_datum.size == node_out_buffer_datum.size);

                for (uint32_t i = 0 + tid; i < work_size; i += nthreads)
                {
                    index_t node = work_source.get_work(i);
			//if(node == 0)    printf("node: %d value:%d inbuffer:%d outbuffer:%d\n",node,node_value_datum[node],node_in_buffer_datum[node],node_out_buffer_datum[node]);
                    const auto pair = app_inst.CombineValueBuffer(node,
                                                                  node_value_datum.get_item_ptr(node),
                                                                  node_in_buffer_datum.get_item_ptr(node));
                    // We always write out buffer returned by "CombineValueBuffer"
		    
		    
                    node_out_buffer_datum[node] = pair.first;
		    //if(node == 0) printf("node: %d value:%d inbuffer:%d outbuffer:%d\n",node,node_value_datum[node],node_in_buffer_datum[node],node_out_buffer_datum[node]);
                    if (pair.second)
                    {
		        //if(node == 0) printf("right\n");
                        work_target.append(node);
                    }
                }
            }

            template<typename TAppInst,
                    typename WorkSource,
                    template<typename> class GraphDatum,
                    typename TValue,
                    typename TBuffer>
            __global__
            void CombineValueBuffer(TAppInst app_inst,
                                    WorkSource work_source,
                                    GraphDatum<TValue> node_value_datum,
                                    GraphDatum<TBuffer> node_in_buffer_datum,
                                    GraphDatum<TBuffer> node_out_buffer_datum)
            {
                uint32_t tid = TID_1D;
                uint32_t nthreads = TOTAL_THREADS_1D;
                uint32_t work_size = work_source.get_size();

                assert(node_value_datum.size == node_in_buffer_datum.size);
                assert(node_value_datum.size == node_out_buffer_datum.size);

                for (uint32_t i = 0 + tid; i < work_size; i += nthreads)
                {
                    index_t node = work_source.get_work(i);

                    const auto pair = app_inst.CombineValueBuffer(node,
                                                                  node_value_datum.get_item_ptr(node),
                                                                  node_in_buffer_datum.get_item_ptr(node));
                    // We always write out buffer returned by "CombineValueBuffer"
                    node_out_buffer_datum[node] = pair.first;
                }                         
            }

            template<typename TAppInst,
                    typename WorkSource,
                    template<typename> class GraphDatum,
                    typename TBuffer>
            __device__ __forceinline__
            void CountActiveNodes(TAppInst app_inst,
                                  WorkSource work_source,
                                  GraphDatum<TBuffer> node_buffer_datum,
                                  uint32_t *p_active_count)
            {
                uint32_t tid = TID_1D;
                uint32_t nthreads = TOTAL_THREADS_1D;
                uint32_t work_size = work_source.get_size();
                uint32_t local_count = 0;

                for (uint32_t i = 0 + tid; i < work_size; i += nthreads)
                {
                    uint32_t node = work_source.get_work(i);

                    if (app_inst.IsActiveNode(node, node_buffer_datum[node]))
                    {
                        local_count++;
                    }
                }

                atomicAdd(p_active_count, local_count);
            }
        }
    }
}
#endif //HYBRID_KERNEL_COMMON_H
