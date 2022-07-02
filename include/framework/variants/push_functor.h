// ----------------------------------------------------------------
// SEP-Graph: Finding Shortest Execution Paths for Graph Processing under a Hybrid Framework on GPU
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE
// in the root directory of this source distribution.
// ----------------------------------------------------------------
#ifndef HYBRID_PUSH_FUNCTOR_H
#define HYBRID_PUSH_FUNCTOR_H

#include <groute/device/queue.cuh>
#include <groute/device/bitmap_impls.h>
#include <framework/variants/api.cuh>

namespace sepgraph
{
    namespace kernel
    {
        template <typename TBuffer>
        struct Payload
        {
            index_t m_src;
            TBuffer m_buffer_to_push;
            //https://stackoverflow.com/questions/33978185/default-constructor-cannot-be-referenced-in-visual-studio-2015
            //X is a union-like class that has a variant member with a non-trivial default constructor,
            //            __device__
            //            Payload()
            //            {
            //
            //            }
            //
            //            __device__ __forceinline__
            //            Payload(index_t src,
            //                    TBuffer buffer_to_push) : m_src(src),
            //                                              m_buffer_to_push(buffer_to_push)
            //            {
            //
            //            }
        };

        template <typename TAppInst,
                  typename TCSRGraph,
                  template <typename> class GraphDatum,
                  typename TBuffer,
                  typename TWeight>
        struct PushFunctor
        {
            typedef groute::dev::Queue<index_t> TWorkTarget;
            TAppInst m_app_inst;
            TWorkTarget m_work_target_low;
            TWorkTarget m_work_target_high;
            TCSRGraph m_csr_graph;
            GraphDatum<TBuffer> m_buffer_array;
            GraphDatum<TWeight> m_weight_array;
            TBuffer m_current_priority;
            bool m_data_driven;
            bool m_priority;
            bool m_weighted;

            __device__
            PushFunctor()
            {
            }
            /**
             * Async+Push+TD
             * @param app_inst
             * @param csr_graph
             * @param buffer_array
             * @param weight_array
             */
            __device__ __forceinline__
            PushFunctor(TAppInst app_inst,
                        TCSRGraph csr_graph,
                        GraphDatum<TBuffer> buffer_array,
                        GraphDatum<TWeight> weight_array) : m_app_inst(app_inst),
                                                            m_work_target_low(nullptr, nullptr, 0),
                                                            m_work_target_high(nullptr, nullptr, 0),
                                                            m_current_priority(0),
                                                            m_csr_graph(csr_graph),
                                                            m_buffer_array(buffer_array),
                                                            m_weight_array(weight_array),
                                                            m_data_driven(false),
                                                            m_priority(false)
            {
                m_weighted = m_weight_array.size > 0;
            }

            /**
             * Async+Push+DD+[Priority]
             * @param app_inst
             * @param work_target_low
             * @param work_target_high
             * @param current_priority
             * @param csr_graph
             * @param buffer_array
             * @param weight_array
             */
            __device__ __forceinline__
            PushFunctor(TAppInst app_inst,
                        TWorkTarget work_target_low,
                        TWorkTarget work_target_high,
                        TBuffer current_priority,
                        TCSRGraph csr_graph,
                        GraphDatum<TBuffer> buffer_array,
                        GraphDatum<TWeight> weight_array) : m_app_inst(app_inst),
                                                            m_work_target_low(work_target_low),
                                                            m_work_target_high(work_target_high),
                                                            m_current_priority(current_priority),
                                                            m_csr_graph(csr_graph),
                                                            m_buffer_array(buffer_array),
                                                            m_weight_array(weight_array),
                                                            m_data_driven(true)
            {
                m_weighted = m_weight_array.size > 0;
                m_priority = work_target_low != work_target_high;
            }

            __device__ __forceinline__ bool operator()(index_t edge, Payload<TBuffer> meta_data)
            {
                index_t dst = m_csr_graph.edge_dest(edge);
                TBuffer buffer_to_push = meta_data.m_buffer_to_push;
                int status;
                bool accumulate_success;
                bool continue_push;

                if (m_weighted)
                {
                    status = m_app_inst.AccumulateBuffer(meta_data.m_src,
                                                         dst,
                                                         m_weight_array[edge],
                                                         m_buffer_array.get_item_ptr(dst),
                                                         buffer_to_push);
                }
                else
                {
                    status = m_app_inst.AccumulateBuffer(meta_data.m_src,
                                                         dst,
                                                         m_buffer_array.get_item_ptr(dst),
                                                         buffer_to_push);
                }

                continue_push = (status == m_app_inst.ACCUMULATE_SUCCESS_CONTINUE ||
                                 status == m_app_inst.ACCUMULATE_FAILURE_CONTINUE);
                accumulate_success = (status == m_app_inst.ACCUMULATE_SUCCESS_BREAK ||
                                      status == m_app_inst.ACCUMULATE_SUCCESS_CONTINUE);

                if (m_data_driven && accumulate_success)
                {
                    if (m_priority)
                    {
                        if (m_app_inst.IsHighPriority(m_current_priority, m_buffer_array[dst]))
                            m_work_target_high.append(dst);
                        else
                            m_work_target_low.append(dst);
                    }
                    else
                    {
                        m_work_target_high.append(dst);
                    }
                }

                return continue_push;
            }
        };

        template <typename TAppInst,
                  typename TCSRGraph,
                  template <typename> class GraphDatum,
                  typename TBuffer,
                  typename TWeight>
        struct PushFunctorDB
        {
            typedef groute::dev::Queue<index_t> TWorkTarget;
            TAppInst m_app_inst;
            TWorkTarget m_work_target_low;
            TWorkTarget m_work_target_high;
            TCSRGraph m_csr_graph;
            GraphDatum<TBuffer> m_buffer_array;
            GraphDatum<TWeight> m_weight_array;
            TBuffer m_current_priority;
            bool m_data_driven;
            bool m_priority;
            bool m_weighted;
            BitmapDeviceObject m_out_active_high;

            __device__
            PushFunctorDB()
            {
            }
            /**
             * Async+Push+TD
             * @param app_inst
             * @param csr_graph
             * @param buffer_array
             * @param weight_array
             */
            __device__ __forceinline__
            PushFunctorDB(TAppInst app_inst,
                          TCSRGraph csr_graph,
                          GraphDatum<TBuffer> buffer_array,
                          GraphDatum<TWeight> weight_array,
                          BitmapDeviceObject out_active) : m_app_inst(app_inst),
                                                           m_work_target_low(nullptr, nullptr, 0),
                                                           m_work_target_high(nullptr, nullptr, 0),
                                                           m_current_priority(0),
                                                           m_csr_graph(csr_graph),
                                                           m_buffer_array(buffer_array),
                                                           m_weight_array(weight_array),
                                                           m_data_driven(true),
                                                           m_priority(false),
                                                           m_out_active_high(out_active)
            {
                m_weighted = m_weight_array.size > 0;
            }

            __device__ __forceinline__ bool operator()(uint64_t edge, Payload<TBuffer> meta_data)
            {
                index_t dst = m_csr_graph.edge_dest(edge);
                TBuffer buffer_to_push = meta_data.m_buffer_to_push;

                if (m_weighted)
                {
                    m_app_inst.AccumulateBuffer(meta_data.m_src,
                                                         dst,
                                                         m_weight_array[edge],
                                                         m_buffer_array.get_item_ptr(dst),
                                                         buffer_to_push);
                }
                else
                {
		    
                   m_app_inst.AccumulateBuffer(meta_data.m_src,
                                                         dst,
                                                         m_buffer_array.get_item_ptr(dst),
                                                         buffer_to_push);
                }


                return true;
            }
        };

        template <typename TAppInst,
                  typename TCSRGraph,
                  template <typename> class GraphDatum,
                  typename TBuffer,
                  typename TWeight>
        struct PushFunctorDB_COM
        {
            typedef groute::dev::Queue<index_t> TWorkTarget;
            TAppInst m_app_inst;
            TWorkTarget m_work_target_low;
            TWorkTarget m_work_target_high;
            TCSRGraph m_csr_graph;
            GraphDatum<TBuffer> m_buffer_array;
            GraphDatum<TWeight> m_weight_array;
            TBuffer m_current_priority;
            bool m_data_driven;
            bool m_priority;
            bool m_weighted;
            BitmapDeviceObject m_out_active_high;

            __device__
            PushFunctorDB_COM()
            {
            }
            /**
             * Async+Push+TD
             * @param app_inst
             * @param csr_graph
             * @param buffer_array
             * @param weight_array
             */
            __device__ __forceinline__
            PushFunctorDB_COM(TAppInst app_inst,
                          TCSRGraph csr_graph,
                          GraphDatum<TBuffer> buffer_array,
                          GraphDatum<TWeight> weight_array,
                          BitmapDeviceObject out_active) : m_app_inst(app_inst),
                                                           m_work_target_low(nullptr, nullptr, 0),
                                                           m_work_target_high(nullptr, nullptr, 0),
                                                           m_current_priority(0),
                                                           m_csr_graph(csr_graph),
                                                           m_buffer_array(buffer_array),
                                                           m_weight_array(weight_array),
                                                           m_data_driven(true),
                                                           m_priority(false),
                                                           m_out_active_high(out_active)
            {
                m_weighted = m_weight_array.size > 0;
            }

            __device__ __forceinline__ bool operator()(uint64_t edge, Payload<TBuffer> meta_data)
            {
                index_t dst =  m_csr_graph.edge_dest(edge);
                TBuffer buffer_to_push = meta_data.m_buffer_to_push;

                if (m_weighted)
                {
                    m_app_inst.AccumulateBuffer(meta_data.m_src,
                                                         dst,
                                                         m_weight_array[edge],
                                                         m_buffer_array.get_item_ptr(dst),
                                                         buffer_to_push);
                }
                else
                {
            
                   m_app_inst.AccumulateBuffer(meta_data.m_src,
                                                         dst,
                                                         m_buffer_array.get_item_ptr(dst),
                                                         buffer_to_push);
                }


                return true;
            }
        };

    } // namespace kernel
} // namespace sepgraph
#endif //HYBRID_PUSH_FUNCTOR_H
