// ----------------------------------------------------------------
// SEP-Graph: Finding Shortest Execution Paths for Graph Processing under a Hybrid Framework on GPU
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE
// in the root directory of this source distribution.
// ----------------------------------------------------------------
#ifndef HYBRID_PULL_FUNCTOR_H
#define HYBRID_PULL_FUNCTOR_H

#include <groute/device/bitmap_impls.h>
#include <framework/variants/api.cuh>

namespace sepgraph
{
    namespace kernel
    {
        template<typename TAppInst,
                typename CSCGraph,
                template<typename> class GraphDatum,
                typename TBuffer,
                typename TWeight>
        struct PullFunctor
        {
            TAppInst m_app_inst;
            BitmapDeviceObject m_in_active;
            BitmapDeviceObject m_out_active_low;
            BitmapDeviceObject m_out_active_high;
            CSCGraph m_csc_graph;
            GraphDatum<TBuffer> m_in_buffer_array;
            GraphDatum<TBuffer> m_out_buffer_array;
            GraphDatum<TWeight> m_weight_array;
            TBuffer m_current_priority;
            bool m_data_driven;
            bool m_priority;
            bool m_weighted;



            /**
             * Async+Pull+DD+[Priority]
             * @param app_inst
             * @param in_active
             * @param out_active
             * @param csc_graph
             * @param buffer_array
             * @param weight_array
             */
            __device__
            PullFunctor(TAppInst app_inst,
                        BitmapDeviceObject in_active,
                        BitmapDeviceObject out_active_low,
                        BitmapDeviceObject out_active_high,
                        TBuffer current_priority,
                        CSCGraph csc_graph,
                        GraphDatum<TBuffer> buffer_array,
                        GraphDatum<TWeight> weight_array) :
                    m_app_inst(app_inst),
                    m_in_active(in_active),
                    m_out_active_low(out_active_low),
                    m_out_active_high(out_active_high),
                    m_current_priority(current_priority),
                    m_csc_graph(csc_graph),
                    m_in_buffer_array(buffer_array),
                    m_out_buffer_array(buffer_array),
                    m_weight_array(weight_array),
                    m_data_driven(true)
            {
                m_weighted = weight_array.size > 0;
                m_priority = out_active_low != out_active_high;
            }

            /**
             * Sync+Pull+DD
             * @param app_inst
             * @param in_active
             * @param out_active
             * @param csc_graph
             * @param buffer_array
             * @param weight_array
             */
            __device__
            PullFunctor(TAppInst app_inst,
                        BitmapDeviceObject in_active,
                        BitmapDeviceObject out_active,
                        CSCGraph csc_graph,
                        GraphDatum<TBuffer> in_buffer_array,
                        GraphDatum<TBuffer> out_buffer_array,
                        GraphDatum<TWeight> weight_array) :
                    m_app_inst(app_inst),
                    m_in_active(in_active),
                    m_out_active_low(out_active),
                    m_out_active_high(out_active),
                    m_csc_graph(csc_graph),
                    m_in_buffer_array(in_buffer_array),
                    m_out_buffer_array(out_buffer_array),
                    m_weight_array(weight_array),
                    m_current_priority(0),
                    m_data_driven(true),
                    m_priority(false)
            {
                m_weighted = weight_array.size > 0;
            }

            /**
             * Sync+Pull+TD
             * @param app_inst
             * @param csc_graph
             * @param in_buffer_array
             * @param out_buffer_array
             * @param weight_array
             */
            __device__
            PullFunctor(TAppInst app_inst,
                        CSCGraph csc_graph,
                        GraphDatum<TBuffer> in_buffer_array,
                        GraphDatum<TBuffer> out_buffer_array,
                        GraphDatum<TWeight> weight_array) :
                    m_app_inst(app_inst),
                    m_in_active(nullptr, 0, nullptr),
                    m_out_active_low(nullptr, 0, nullptr),
                    m_out_active_high(nullptr, 0, nullptr),
                    m_csc_graph(csc_graph),
                    m_in_buffer_array(in_buffer_array),
                    m_out_buffer_array(out_buffer_array),
                    m_weight_array(weight_array),
                    m_data_driven(false),
                    m_priority(false)
            {
                m_weighted = weight_array.size > 0;
            }

            /***
             * dest Pull from source
             * @param edge
             * @param dst
             * @return true to continue pull from neighbors, false to give up pull
             */
            __device__ __forceinline__
            bool operator()(index_t edge, index_t dst)
            {
                index_t src = m_csc_graph.edge_src(edge);
		//printf("%d %d debug3\n", dst, src);
                int status;
                bool accumulate_success = false;
                bool continue_pull = true;

                
                    TBuffer new_buffer = m_in_buffer_array[src];

                    if (m_weighted)
                    {
                        status = m_app_inst.AccumulateBuffer(src,
                                                             dst,
                                                             m_weight_array.get_item(edge),
                                                             m_out_buffer_array.get_item_ptr(dst),
                                                             new_buffer);
                    }
                    else
                    {
                        status = m_app_inst.AccumulateBuffer(src,
                                                             dst,
                                                             m_out_buffer_array.get_item_ptr(dst),
                                                             new_buffer);
                    }

                    continue_pull = (status == m_app_inst.ACCUMULATE_SUCCESS_CONTINUE ||
                                     status == m_app_inst.ACCUMULATE_FAILURE_CONTINUE);
                    accumulate_success = (status == m_app_inst.ACCUMULATE_SUCCESS_BREAK ||
                                          status == m_app_inst.ACCUMULATE_SUCCESS_CONTINUE);

                    if (m_data_driven && accumulate_success)
                    {
                        if (m_priority)
                        {
                            if (m_app_inst.IsHighPriority(m_current_priority, m_in_buffer_array[src]))
                                m_out_active_high.set_bit_atomic(dst);
                            else
                                m_out_active_low.set_bit_atomic(dst);
                        }
                        else
                        {
                            m_out_active_high.set_bit_atomic(dst);
                        }
                    }
                

                return continue_pull;
            }
        };
	
	template<typename TAppInst,
                typename CSCGraph,
                template<typename> class GraphDatum,
                typename TBuffer,
                typename TWeight>
        struct PullFunctorDB
        {
            TAppInst m_app_inst;
            BitmapDeviceObject m_in_active;
            BitmapDeviceObject m_out_active_low;
            BitmapDeviceObject m_out_active_high;
            CSCGraph m_csc_graph;
            GraphDatum<TBuffer> m_in_buffer_array;
            GraphDatum<TBuffer> m_out_buffer_array;
            GraphDatum<TWeight> m_weight_array;
            TBuffer m_current_priority;
            bool m_data_driven;
            bool m_priority;
            bool m_weighted;



            /**
             * Async+Pull+DD+[Priority]
             * @param app_inst
             * @param in_active
             * @param out_active
             * @param csc_graph
             * @param buffer_array
             * @param weight_array
             */
            __device__
            PullFunctorDB(TAppInst app_inst,
                        BitmapDeviceObject in_active,
                        BitmapDeviceObject out_active_low,
                        BitmapDeviceObject out_active_high,
                        TBuffer current_priority,
                        CSCGraph csc_graph,
                        GraphDatum<TBuffer> buffer_array,
                        GraphDatum<TWeight> weight_array) :
                    m_app_inst(app_inst),
                    m_in_active(in_active),
                    m_out_active_low(out_active_low),
                    m_out_active_high(out_active_high),
                    m_current_priority(current_priority),
                    m_csc_graph(csc_graph),
                    m_in_buffer_array(buffer_array),
                    m_out_buffer_array(buffer_array),
                    m_weight_array(weight_array),
                    m_data_driven(true)
            {
                m_weighted = weight_array.size > 0;
                m_priority = out_active_low != out_active_high;
            }

            /**
             * Sync+Pull+DD
             * @param app_inst
             * @param in_active
             * @param out_active
             * @param csc_graph
             * @param buffer_array
             * @param weight_array
             */
            __device__
            PullFunctorDB(TAppInst app_inst,
                        BitmapDeviceObject in_active,
                        BitmapDeviceObject out_active,
                        CSCGraph csc_graph,
                        GraphDatum<TBuffer> in_buffer_array,
                        GraphDatum<TBuffer> out_buffer_array,
                        GraphDatum<TWeight> weight_array) :
                    m_app_inst(app_inst),
                    m_in_active(in_active),
                    m_out_active_low(out_active),
                    m_out_active_high(out_active),
                    m_csc_graph(csc_graph),
                    m_in_buffer_array(in_buffer_array),
                    m_out_buffer_array(out_buffer_array),
                    m_weight_array(weight_array),
                    m_current_priority(0),
                    m_data_driven(true),
                    m_priority(false)
            {
                m_weighted = weight_array.size > 0;
            }

            /**
             * Sync+Pull+TD
             * @param app_inst
             * @param csc_graph
             * @param in_buffer_array
             * @param out_buffer_array
             * @param weight_array
             */
            __device__
            PullFunctorDB(TAppInst app_inst,
                        CSCGraph csc_graph,
                        GraphDatum<TBuffer> in_buffer_array,
                        GraphDatum<TBuffer> out_buffer_array,
                        GraphDatum<TWeight> weight_array,
			BitmapDeviceObject m_out_active_high) :
                    m_app_inst(app_inst),
                    m_in_active(nullptr, 0, nullptr),
                    m_out_active_low(nullptr, 0, nullptr),
                    m_out_active_high(m_out_active_high),
                    m_csc_graph(csc_graph),
                    m_in_buffer_array(in_buffer_array),
                    m_out_buffer_array(out_buffer_array),
                    m_weight_array(weight_array),
                    m_data_driven(false),
                    m_priority(false)
            {
                m_weighted = weight_array.size > 0;
            }

            /***
             * dest Pull from source
             * @param edge
             * @param dst
             * @return true to continue pull from neighbors, false to give up pull
             */
            __device__ __forceinline__
            bool operator()(index_t edge, index_t dst)
            {
	      
                index_t src = m_csc_graph.edge_src(edge);
		//printf("%d %d debug3\n", dst, src);
                int status;
                bool accumulate_success = false;
                bool continue_pull = true;

                
                    TBuffer new_buffer = m_in_buffer_array[src];

                    if (m_weighted)
                    {
                        status = m_app_inst.AccumulateBuffer(src,
                                                             dst,
                                                             m_weight_array.get_item(edge),
                                                             m_out_buffer_array.get_item_ptr(dst),
                                                             new_buffer);
                    }
                    else
                    {
                        status = m_app_inst.AccumulateBuffer(src,
                                                             dst,
                                                             m_out_buffer_array.get_item_ptr(dst),
                                                             new_buffer);
                    }

                    continue_pull = (status == m_app_inst.ACCUMULATE_SUCCESS_CONTINUE ||
                                     status == m_app_inst.ACCUMULATE_FAILURE_CONTINUE);
                    accumulate_success = (status == m_app_inst.ACCUMULATE_SUCCESS_BREAK ||
                                          status == m_app_inst.ACCUMULATE_SUCCESS_CONTINUE);

                    /*if (m_data_driven && accumulate_success)
                    {
                        if (m_priority)
                        {
                            if (m_app_inst.IsHighPriority(m_current_priority, m_in_buffer_array[src]))
                                m_out_active_high.set_bit_atomic(dst);
                            else
                                m_out_active_low.set_bit_atomic(dst);
                        }
                        else
                        {
                            m_out_active_high.set_bit_atomic(dst);
                        }
                    }*/
                

                return continue_pull;
            }
        };
    }
}

#endif //HYBRID_PULL_FUNCTOR_H
