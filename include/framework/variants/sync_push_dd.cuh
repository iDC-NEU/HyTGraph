// ----------------------------------------------------------------
// SEP-Graph: Finding Shortest Execution Paths for Graph Processing under a Hybrid Framework on GPU
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE
// in the root directory of this source distribution.
// ----------------------------------------------------------------
#ifndef HYBRID_SYNC_PUSH_DD_H
#define HYBRID_SYNC_PUSH_DD_H
#define BLOCK_SIZE 1024
#include <utils/cuda_utils.h>
#include <groute/device/cta_scheduler_hybrid.cuh>
#include <framework/variants/push_functor.h>
#include <framework/variants/pull_functor.h>
#include <framework/variants/common.cuh>
#include <framework/common.h>

namespace sepgraph
{
    namespace kernel
    {
        namespace sync_push_dd
        {
            using sepgraph::common::LoadBalancing;

            template<typename TAppInst,
                    typename WorkSource,
                    typename WorkTarget,
                    typename CSRGraph,
                    template<typename> class GraphDatum,
                    typename TValue,
                    typename TBuffer,
                    typename TWeight>
            __forceinline__ __device__
            void Relax(TAppInst app_inst,
		       index_t seg_snode,
			   index_t seg_enode,
			   uint64_t seg_sedge_csr,
		       bool zcflag,
                       WorkSource work_source,
                       WorkTarget work_target_low,
                       WorkTarget work_target_high,
                       TBuffer current_priority,
                       CSRGraph csr_graph,
                       GraphDatum<TValue> node_value_datum,
                       GraphDatum<TBuffer> node_buffer_datum,
                       GraphDatum<TWeight> edge_weight_datum)
            {
                uint32_t tid = TID_1D;
                uint32_t nthreads = TOTAL_THREADS_1D;
                uint32_t work_size = work_source.get_size();
		
                PushFunctor<TAppInst, CSRGraph, GraphDatum, TBuffer, TWeight>
                        push_functor(app_inst,
                                     work_target_low,
                                     work_target_high,
                                     current_priority,
                                     csr_graph,
                                     node_buffer_datum,
                                     edge_weight_datum);
		
		 
		
               /*for (int i = 0 + tid; i < work_size; i += nthreads)
                {
                    index_t node = work_source.get_work(i);
		            if(node >= seg_snode && node < seg_enode){
		                auto pair = app_inst.CombineValueBuffer(node,
                                                            node_value_datum.get_item_ptr(node),
                                                            node_buffer_datum.get_item_ptr(node));
				 printf("node: %d buffer_to_push: %d\n", node, pair.first);
		                if (pair.second)
                        {
                            Payload<TBuffer> payload;
                            payload.m_src = node;
                            payload.m_buffer_to_push = pair.first;
			    printf("node: %d buffer_to_push: %d\n", node, pair.first);
			                if(zcflag == false){
                                for (index_t edge = csr_graph.begin_edge(node) - seg_sedge_csr, end_edge = csr_graph.end_edge(node) - seg_sedge_csr; edge < end_edge; edge++){
                                    if (!push_functor(edge, payload)){
                                        break;
                                    }
                                }
			                }else{
			                    for (index_t edge = csr_graph.begin_edge(node), end_edge = csr_graph.end_edge(node); edge < end_edge; edge++){
                                    if (!push_functor(edge, payload)){
                                        break;
                                    }
                                }
			                }
                        }
		            }
		        }*/
            }
                    template<typename TAppInst,
                    typename WorkSource,
                    typename CSRGraph,
                    template<typename> class GraphDatum,
                    typename TValue,
                    typename TBuffer,
                    typename TWeight>  
            __forceinline__ __device__
            void RelaxDB(TAppInst app_inst,
		       index_t seg_snode,
			   index_t seg_enode,
			   uint64_t seg_sedge_csr,
		       bool zcflag,
                       WorkSource work_source,
                       CSRGraph csr_graph,
                       GraphDatum<TValue> node_value_datum,
                       GraphDatum<TBuffer> node_buffer_datum,
                       GraphDatum<TWeight> edge_weight_datum,
                       BitmapDeviceObject out_active)
            {
                uint32_t tid = TID_1D;
                uint32_t nthreads = TOTAL_THREADS_1D;
                uint32_t work_size = work_source.get_size();
		
                PushFunctorDB<TAppInst, CSRGraph, GraphDatum, TBuffer, TWeight>
                             push_functor(app_inst,
                             csr_graph,
                             node_buffer_datum,
                             edge_weight_datum,
                             out_active);

		
               for (int i = 0 + tid; i < work_size; i += nthreads)
                {
                    index_t node = work_source.get_work(i);
		                auto pair = app_inst.CombineValueBuffer(node,
                                                            node_value_datum.get_item_ptr(node),
                                                            node_buffer_datum.get_item_ptr(node));
		                if (pair.second)
                        {
                            
                            Payload<TBuffer> payload;
                            payload.m_src = node;
                            payload.m_buffer_to_push = pair.first;
			    
			    //printf("node: %d buffer_to_push: %d\n", node, pair.first);
			                if(zcflag == false){
                                for (index_t edge = csr_graph.begin_edge(node) - seg_sedge_csr, end_edge = csr_graph.end_edge(node) - seg_sedge_csr; edge < end_edge; edge++){
                                    if (!push_functor(edge, payload)){
                                        break;
                                    }
                                    //printf("%d %d %d\n",node, csr_graph.begin_edge(node)-seg_sedge_csr, end_edge-csr_graph.begin_edge(node));
                                }
			                }else{
			                    for (index_t edge = csr_graph.begin_edge(node), end_edge = csr_graph.end_edge(node); edge < end_edge; edge++){
                                    if (!push_functor(edge, payload)){
                                        break;
                                    }
                                    //printf("%d %d %d\n",node, csr_graph.begin_edge(node), end_edge-csr_graph.begin_edge(node));
                                }
                            }
                           
                        }
		            
		        }
            }

           template<LoadBalancing LB,
                    bool enable_priority,
                    typename TAppInst,
                    typename WorkSource,
                    typename CSRGraph,
                    template<typename> class GraphDatum,
                    typename TValue,
                    typename TBuffer,
                    typename TWeight>
            __forceinline__ __device__
            void RelaxCTADB(TAppInst app_inst,
                            index_t seg_snode,
                            index_t seg_enode,
                            uint64_t seg_sedge_csr,
                            bool zcflag,
                          WorkSource work_source,
                          const CSRGraph csr_graph,
                          GraphDatum<TValue> node_value_datum,
                          GraphDatum<TBuffer> node_buffer_datum,
                          GraphDatum<TWeight> edge_weight_datum,
                          TBuffer current_priority,
                          BitmapDeviceObject out_active,
                          BitmapDeviceObject in_active)
            {
                const uint32_t tid = TID_1D;
                //if(tid==0)printf("RelaxCTADB\n");
                const uint32_t nthreads = TOTAL_THREADS_1D;
                const uint32_t work_size = work_source.get_size();
                const uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;
                PushFunctorDB<TAppInst, CSRGraph, GraphDatum, TBuffer, TWeight>
                        push_functor(app_inst, csr_graph, node_buffer_datum, edge_weight_datum,out_active);

                        for (uint32_t i = 0 + tid; i < work_size_rup; i += nthreads)
                       {
                            groute::dev::np_local<Payload<TBuffer>> np_local = {0, 0};
        
                            if (tid < work_size)
                            {
                                    const index_t node = work_source.get_work(tid);
				                    //printf("node:%d\n",node);
                                    const auto pair = app_inst.CombineValueBuffer(node,
                                                                                  node_value_datum.get_item_ptr(node),
                                                                                  node_buffer_datum.get_item_ptr(node));
                                    //out_active.set_bit_atomic(node);
                                    // Value changed means validate combine, we need push the buffer to neighbors
                                    if (pair.second)
                                    {       
                                        np_local.start = csr_graph.begin_edge(node);
                                        np_local.size = csr_graph.end_edge(node) - np_local.start; // out-degree
                                        if(!zcflag){
                                            np_local.start -= seg_sedge_csr;
                                        } 
                                        Payload<TBuffer> payload;
                                        payload.m_src = node;
                                        payload.m_buffer_to_push = pair.first;
                                        np_local.meta_data = payload;
                                        //printf("%d %d %d\n",node, np_local.start, np_local.size);
                                        // if(node==10767||node==785951||node==828471||node==851670)
                                        //     for(int j=np_local.start;j<np_local.start+np_local.size;j++){
                                        //         printf("##%d %d\n",node,csr_graph.edge_dest(j));
                                        //     }
                                    }
                                
				
                             }
        
                            switch (LB)
                            {
                                case LoadBalancing::COARSE_GRAINED:
                                    groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_COARSE_GRAINED>::template
                                    schedule(np_local, push_functor,zcflag);
                                    break;
                                case LoadBalancing::FINE_GRAINED:
                                    groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_FINE_GRAINED>::template
                                    schedule(np_local, push_functor,zcflag);
                                    break;
                                case LoadBalancing::HYBRID:
                                    groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_HYBRID>::template
                                    schedule(np_local, push_functor,zcflag);
                                    break;
                                default:
                                    assert(false);
                            }
                        }
            }

            template<LoadBalancing LB,
                    bool enable_priority,
                    typename TAppInst,
                    typename WorkSource,
                    typename CSRGraph,
                    template<typename> class GraphDatum,
                    typename TValue,
                    typename TBuffer,
                    typename TWeight>
            __forceinline__ __device__
            void RelaxCTADB_COM(TAppInst app_inst,
                                WorkSource work_source,
                                const CSRGraph csr_graph,
                                GraphDatum<TValue> node_value_datum,
                                GraphDatum<TBuffer> node_buffer_datum,
                                GraphDatum<TWeight> edge_weight_datum,
                                TBuffer current_priority,
                                BitmapDeviceObject out_active,
                                BitmapDeviceObject in_active)
            {
                const uint32_t tid = TID_1D;
                const uint32_t nthreads = TOTAL_THREADS_1D;
                const uint32_t work_size = work_source.get_size();
                const uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;
                PushFunctorDB_COM<TAppInst, CSRGraph, GraphDatum, TBuffer, TWeight>
                        push_functor(app_inst, csr_graph, node_buffer_datum, edge_weight_datum,out_active);

                        for (uint32_t i = 0 + tid; i < work_size_rup; i += nthreads)
                       {
                            groute::dev::np_local<Payload<TBuffer>> np_local = {0, 0};
        
                            if (tid < work_size)
                            {
                                    const index_t node = csr_graph.subgraph_activenode[tid];
                                    //printf("node:%d\n",node);
                                    const auto pair = app_inst.CombineValueBuffer(node,
                                                                                  node_value_datum.get_item_ptr(node),
                                                                                  node_buffer_datum.get_item_ptr(node));
                                    //out_active.set_bit_atomic(node);
                                    // Value changed means validate combine, we need push the buffer to neighbors
                                    if (pair.second)
                                    {       
                                        np_local.start = csr_graph.subgraph_rowstart[tid];
                                        np_local.size = csr_graph.subgraph_rowstart[tid + 1] - np_local.start; // out-degree
                                        Payload<TBuffer> payload;
                                        payload.m_src = node;
                                        payload.m_buffer_to_push = pair.first;
                                        np_local.meta_data = payload;
                                        
                                        //printf("%d %d %d\n",node, np_local.start, np_local.size);
                                        // if(node==10767||node==785951||node==828471||node==851670)
                                        //     for(int j=np_local.start;j<np_local.start+np_local.size;j++){
                                        //         printf("##%d %d\n",node,csr_graph.edge_dest(j));
                                        //     }
                                    }
                                
                
                             }
        
                            switch (LB)
                            {
                                case LoadBalancing::COARSE_GRAINED:
                                    groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_COARSE_GRAINED>::template
                                    schedule(np_local, push_functor,false);
                                    break;
                                case LoadBalancing::FINE_GRAINED:
                                    groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_FINE_GRAINED>::template
                                    schedule(np_local, push_functor,false);
                                    break;
                                case LoadBalancing::HYBRID:
                                    groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_HYBRID>::template
                                    schedule(np_local, push_functor,false);
                                    break;
                                default:
                                    assert(false);
                            }
                        }
            }


            template<LoadBalancing LB,
                    typename TAppInst,
                    typename WorkSource,
                    typename WorkTarget,
                    typename CSRGraph,
                    template<typename> class GraphDatum,
                    typename TValue,
                    typename TBuffer,
                    typename TWeight>
            __forceinline__ __device__
            void RelaxCTA(TAppInst app_inst,
                          index_t seg_snode,
                          index_t seg_enode,
                          index_t seg_sedge_csr,
                          bool zcflag,
                          WorkSource work_source,
                          WorkTarget work_target_low,
                          WorkTarget work_target_high,
                          TBuffer current_priority,
                          CSRGraph csr_graph,
                          GraphDatum<TValue> node_value_datum,
                          GraphDatum<TBuffer> node_buffer_datum,
                          GraphDatum<TWeight> edge_weight_datum)
            {
                const uint32_t tid = TID_1D;
                if(tid==0) printf("PUSHDD\n");
                const uint32_t nthreads = TOTAL_THREADS_1D;
                const uint32_t work_size = work_source.get_size();
                const uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;
                PushFunctor<TAppInst, CSRGraph, GraphDatum, TBuffer, TWeight>
                        push_functor(app_inst, work_target_low, work_target_high, current_priority,
                                     csr_graph, node_buffer_datum, edge_weight_datum);

               /* for (int i = 0 + tid; i < work_size_rup; i += nthreads)
                {
                    groute::dev::np_local<Payload<TBuffer>> np_local = {0, 0};
                    index_t node_to_process=seg_enode+1;

                    if (i < work_size)
                    {
                        const index_t node = work_source.get_work(i);
                        node_to_process=node;
                        auto pair = app_inst.CombineValueBuffer(node,
                                                                node_value_datum.get_item_ptr(node),
                                                                node_buffer_datum.get_item_ptr(node));

                        if (pair.second)
                        {
                            if(zcflag){
                                np_local.start = csr_graph.begin_edge(node);
                                np_local.size = csr_graph.end_edge(node) - np_local.start; // out-degree
                            }else{
                                np_local.start = csr_graph.begin_edge(node) - seg_sedge_csr;
                                np_local.size = csr_graph.end_edge(node) - seg_sedge_csr - np_local.start; // out-degree
                            }
                            Payload<TBuffer> payload;
                            payload.m_src = node;
                            payload.m_buffer_to_push = pair.first;
                            np_local.meta_data = payload;
                        }
                    }
                    
                    if(np_local.meta_data.m_src >= seg_snode && np_local.meta_data.m_src < seg_enode){
                        switch (LB)
                        {
                            case LoadBalancing::COARSE_GRAINED:
                                groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_COARSE_GRAINED>::template
                                schedule(np_local, push_functor);
                                break;
                            case LoadBalancing::FINE_GRAINED:
                                groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_FINE_GRAINED>::template
                                schedule(np_local, push_functor);
                                break;
                            case LoadBalancing::HYBRID:
                                groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_HYBRID>::template
                                schedule(np_local, push_functor);
                                break;
                            default:
                                assert(false);
                        }
                    }
                }*/
            }

        template<typename TAppInst,
            typename WorkSource,
            typename WorkTarget,
            typename CSRGraph,
            template<typename> class GraphDatum,
            typename TValue,
            typename TBuffer,
            typename TWeight>
    __forceinline__ __device__
    void Relax_ZC(TAppInst app_inst,
       index_t seg_snode,
       index_t seg_enode,
       uint64_t seg_sedge_csr,
       bool zcflag,
               WorkSource work_source,
               WorkTarget work_target_low,
               WorkTarget work_target_high,
               TBuffer current_priority,
               CSRGraph csr_graph,
               GraphDatum<TValue> node_value_datum,
               GraphDatum<TBuffer> node_buffer_datum,
               GraphDatum<TWeight> edge_weight_datum)
    {
        uint32_t tid = TID_1D;
        uint32_t nthreads = TOTAL_THREADS_1D;
        uint32_t work_size = work_source.get_size();

        PushFunctor<TAppInst, CSRGraph, GraphDatum, TBuffer, TWeight>
                push_functor(app_inst,
                             work_target_low,
                             work_target_high,
                             current_priority,
                             csr_graph,
                             node_buffer_datum,
                             edge_weight_datum);

 

        for (int i = 0 + tid; i < work_size; i += nthreads)
        {
            index_t node = work_source.get_work(i);
            if(node >= seg_snode && node < seg_enode){
                auto pair = app_inst.CombineValueBuffer(node,
                                                    node_value_datum.get_item_ptr(node),
                                                    node_buffer_datum.get_item_ptr(node));
                if (pair.second)
                {
                    Payload<TBuffer> payload;
                    payload.m_src = node;
                    payload.m_buffer_to_push = pair.first;
                    if(zcflag == false){
                        for (index_t edge = csr_graph.begin_edge(node) - seg_sedge_csr, end_edge = csr_graph.end_edge(node) - seg_sedge_csr; edge < end_edge; edge++){
                            if (!push_functor(edge, payload)){
                                break;
                            }
                        }
                    }else{
                        for (index_t edge = csr_graph.begin_edge(node), end_edge = csr_graph.end_edge(node); edge < end_edge; edge++){
                            if (!push_functor(edge, payload)){
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

        template<typename TAppInst,
            typename WorkSource,
            typename WorkTarget,
            typename CSRGraph,
            template<typename> class GraphDatum,
            typename TValue,
            typename TBuffer,
            typename TWeight>
    __forceinline__ __device__
    void Relax_segment(TAppInst app_inst,
       index_t seg_snode,
       index_t seg_enode,
       uint64_t seg_sedge_csr,
       bool zcflag,
               WorkSource work_source,
               WorkTarget work_target_low,
               WorkTarget work_target_high,
               TBuffer current_priority,
               CSRGraph csr_graph,
               GraphDatum<TValue> node_value_datum,
               GraphDatum<TBuffer> node_buffer_datum,
               GraphDatum<TWeight> edge_weight_datum)
    {
        uint32_t tid = TID_1D;
        uint32_t nthreads = TOTAL_THREADS_1D;
        uint32_t work_size = work_source.get_size();

        PushFunctor<TAppInst, CSRGraph, GraphDatum, TBuffer, TWeight>
                push_functor(app_inst,
                             work_target_low,
                             work_target_high,
                             current_priority,
                             csr_graph,
                             node_buffer_datum,
                             edge_weight_datum);

 

        for (int i = 0 + tid; i < work_size; i += nthreads)
        {
            index_t node = work_source.get_work(i);
            if(node >= seg_snode && node < seg_enode){
                auto pair = app_inst.CombineValueBuffer(node,
                                                    node_value_datum.get_item_ptr(node),
                                                    node_buffer_datum.get_item_ptr(node));
                if (pair.second)
                {
                    Payload<TBuffer> payload;
                    payload.m_src = node;
                    payload.m_buffer_to_push = pair.first;
                    for (index_t edge = csr_graph.begin_edge(node) - seg_sedge_csr, end_edge = csr_graph.end_edge(node) - seg_sedge_csr; edge < end_edge; edge++){
                        if (!push_functor(edge, payload)){
                            break;
                        }
                    }
                }
            }
        }
    }


            template<LoadBalancing LB,
                    typename TAppInst,
                    typename WorkSource,
                    typename WorkTarget,
                    typename CSRGraph,
                    template<typename> class GraphDatum,
                    typename TValue,
                    typename TBuffer,
                    typename TWeight>
            __forceinline__ __device__
            void RelaxCTA_legacy(TAppInst app_inst,
                          WorkSource work_source,
                          WorkTarget work_target_low,
                          WorkTarget work_target_high,
                          TBuffer current_priority,
                          CSRGraph csr_graph,
                          GraphDatum<TValue> node_value_datum,
                          GraphDatum<TBuffer> node_buffer_datum,
                          GraphDatum<TWeight> edge_weight_datum)
            {
                const uint32_t tid = TID_1D;
                if(tid==0) printf("PUSHDD\n");
                const uint32_t nthreads = TOTAL_THREADS_1D;
                const uint32_t work_size = work_source.get_size();
                const uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;
                PushFunctor<TAppInst, CSRGraph, GraphDatum, TBuffer, TWeight>
                        push_functor(app_inst, work_target_low, work_target_high, current_priority,
                                     csr_graph, node_buffer_datum, edge_weight_datum);

                for (int i = 0 + tid; i < work_size_rup; i += nthreads)
                {
                    groute::dev::np_local<Payload<TBuffer>> np_local = {0, 0};

                    if (i < work_size)
                    {
                        const index_t node = work_source.get_work(i);
                        auto pair = app_inst.CombineValueBuffer(node,
                                                                node_value_datum.get_item_ptr(node),
                                                                node_buffer_datum.get_item_ptr(node));

                        if (pair.second)
                        {
                            np_local.start = csr_graph.begin_edge(node);
                            np_local.size = csr_graph.end_edge(node) - np_local.start; // out-degree
                            Payload<TBuffer> payload;
                            payload.m_src = node;
                            payload.m_buffer_to_push = pair.first;
                            np_local.meta_data = payload;
                        }
                    }

                    switch (LB)
                    {
                        case LoadBalancing::COARSE_GRAINED:
                            groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_COARSE_GRAINED>::template
                            schedule(np_local, push_functor);
                            break;
                        case LoadBalancing::FINE_GRAINED:
                            groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_FINE_GRAINED>::template
                            schedule(np_local, push_functor);
                            break;
                        case LoadBalancing::HYBRID:
                            groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_HYBRID>::template
                            schedule(np_local, push_functor);
                            break;
                        default:
                            assert(false);
                    }
                }
            }

            template<LoadBalancing LB,
                    typename TAppInst,
                    typename WorkSource,
                    typename WorkTarget,
                    typename CSRGraph,
                    template<typename> class GraphDatum,
                    typename TValue,
                    typename TBuffer,
                    typename TWeight>
            __forceinline__ __device__
            void RelaxCTA_ZC(TAppInst app_inst,
                          index_t seg_snode,
                          index_t seg_enode,
                          index_t seg_sedge_csr,
                          WorkSource work_source,
                          WorkTarget work_target_low,
                          WorkTarget work_target_high,
                          TBuffer current_priority,
                          CSRGraph csr_graph,
                          GraphDatum<TValue> node_value_datum,
                          GraphDatum<TBuffer> node_buffer_datum,
                          GraphDatum<TWeight> edge_weight_datum)
            {
                const uint32_t tid = TID_1D;
                const uint32_t nthreads = TOTAL_THREADS_1D;
                const uint32_t work_size = work_source.get_size();
                const uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;
                PushFunctor<TAppInst, CSRGraph, GraphDatum, TBuffer, TWeight>
                        push_functor(app_inst, work_target_low, work_target_high, current_priority,
                                     csr_graph, node_buffer_datum, edge_weight_datum);

                for (int i = 0 + tid; i < work_size_rup; i += nthreads)
                {
                    groute::dev::np_local<Payload<TBuffer>> np_local = {0, 0};

                    if (i < work_size)
                    {
                        const index_t node = work_source.get_work(i);
                        auto pair = app_inst.CombineValueBuffer(node,
                                                                node_value_datum.get_item_ptr(node),
                                                                node_buffer_datum.get_item_ptr(node));

                        if (pair.second)
                        {
                            np_local.start = csr_graph.begin_edge(node);
                            np_local.size = csr_graph.end_edge(node) - np_local.start; // out-degree
                            Payload<TBuffer> payload;
                            payload.m_src = node;
                            payload.m_buffer_to_push = pair.first;
                            np_local.meta_data = payload;
                        }
                    }

                    switch (LB)
                    {
                        case LoadBalancing::COARSE_GRAINED:
                            groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_COARSE_GRAINED>::template
                            schedule(np_local, push_functor);
                            break;
                        case LoadBalancing::FINE_GRAINED:
                            groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_FINE_GRAINED>::template
                            schedule(np_local, push_functor);
                            break;
                        case LoadBalancing::HYBRID:
                            groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_HYBRID>::template
                            schedule(np_local, push_functor);
                            break;
                        default:
                            assert(false);
                    }
                }
            }

            template<LoadBalancing LB,
                typename TAppInst,
                typename WorkSource,
                typename WorkTarget,
                typename CSRGraph,
                template<typename> class GraphDatum,
                typename TValue,
                typename TBuffer,
                typename TWeight>
            __forceinline__ __device__
            void RelaxCTA_segment(TAppInst app_inst,
                          index_t seg_snode,
                          index_t seg_enode,
                          index_t seg_sedge_csr,
                          WorkSource work_source,
                          WorkTarget work_target_low,
                          WorkTarget work_target_high,
                          TBuffer current_priority,
                          CSRGraph csr_graph,   
                          GraphDatum<TValue> node_value_datum,
                          GraphDatum<TBuffer> node_buffer_datum,
                          GraphDatum<TWeight> edge_weight_datum)
            {
                const uint32_t tid = TID_1D;
                const uint32_t nthreads = TOTAL_THREADS_1D;
                const uint32_t work_size = work_source.get_size();
                const uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;
                PushFunctor<TAppInst, CSRGraph, GraphDatum, TBuffer, TWeight>
                        push_functor(app_inst, work_target_low, work_target_high, current_priority,
                                     csr_graph, node_buffer_datum, edge_weight_datum);

                for (int i = 0 + tid; i < work_size_rup; i += nthreads)
                {
                    groute::dev::np_local<Payload<TBuffer>> np_local = {0, 0};

                    if (i < work_size)
                    {
                        const index_t node = work_source.get_work(i);
                        auto pair = app_inst.CombineValueBuffer(node,
                                                                node_value_datum.get_item_ptr(node),
                                                                node_buffer_datum.get_item_ptr(node));
                        if (pair.second)
                        {
                            np_local.start = csr_graph.begin_edge(node) - seg_sedge_csr;
                            np_local.size = csr_graph.end_edge(node) - seg_sedge_csr - np_local.start; // out-degree
                            Payload<TBuffer> payload;
                            payload.m_src = node;
                            payload.m_buffer_to_push = pair.first;
                            np_local.meta_data = payload;
                        }
                    }

                    switch (LB)
                    {
                        case LoadBalancing::COARSE_GRAINED:
                            groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_COARSE_GRAINED>::template
                            schedule(np_local, push_functor);
                            break;
                        case LoadBalancing::FINE_GRAINED:
                            groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_FINE_GRAINED>::template
                            schedule(np_local, push_functor);
                            break;
                        case LoadBalancing::HYBRID:
                            groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_HYBRID>::template
                            schedule(np_local, push_functor);
                            break;
                        default:
                            assert(false);
                    }   
                }
            }







        }
    }
}
#endif //HYBRID_ASYNC_PUSH_DD_H
