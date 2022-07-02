// ----------------------------------------------------------------
// SEP-Graph: Finding Shortest Execution Paths for Graph Processing under a Hybrid Framework on GPU
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE
// in the root directory of this source distribution.
// ----------------------------------------------------------------
#ifndef HYBRID_ALGO_VARIANTS_H
#define HYBRID_ALGO_VARIANTS_H

#include <cub/grid/grid_barrier.cuh>
#include <framework/common.h>
#include <framework/hybrid_policy.h>
#include <framework/graph_datum.cuh>
#include <framework/variants/driver.cuh>
#include <framework/variants/common.cuh>
#include <framework/variants/pull_functor.h>

namespace sepgraph
{
    namespace engine
    {
        using common::LoadBalancing;
        using common::EngineOptions;

        template<typename TBitmap,
                typename TWorklist>
        void Bitmap2Queue(TBitmap &bitmap,
                          TWorklist &queue,
                          const groute::Stream &stream)
        {
            dim3 grid_dims, block_dims;

            KernelSizing(grid_dims, block_dims, bitmap.GetSize());
            queue.ResetAsync(stream);

            kernel::BitmapToQueue << < grid_dims, block_dims, 0, stream.cuda_stream >> > (bitmap.DeviceObject(),
                    queue.DeviceObject());
            stream.Sync();
        }
	
	template<typename TBitmap,
                typename TWorklist>
	void Bitmap2QueueRange(TBitmap &bitmap,
                          TWorklist &queue,index_t seg_snode, index_t seg_enode,
                          const groute::Stream &stream)
        {
            dim3 grid_dims, block_dims;

            KernelSizing(grid_dims, block_dims, bitmap.GetSize());
            queue.ResetAsync(stream);

            kernel::BitmapToQueueRange << < grid_dims, block_dims, 0, stream.cuda_stream >> > (bitmap.DeviceObject(),
                    queue.DeviceObject(), seg_snode, seg_enode);
            stream.Sync();
        }
        
        
        template<typename TWorklist,
                typename TBitmap>
        void Queue2Bitmap(const TWorklist &queue,
                          TBitmap &bitmap,
                          const groute::Stream &stream)
        {
            dim3 grid_dims, block_dims;

            KernelSizing(grid_dims, block_dims, queue.GetCount(stream));
            bitmap.ResetAsync(stream);

            // TODO use SMs blocks and 256 threads/block
            kernel::QueueToBitmap << < grid_dims, block_dims, 0, stream.cuda_stream >> > (queue.DeviceObject(),
                    bitmap.DeviceObject());
            stream.Sync();
        }


        template<typename TAppInst, typename TGraphDatum>
        void RebuildArrayWorklist(TAppInst &app_inst,
                                  TGraphDatum &graph_datum,
                                  const groute::Stream &stream,
				  index_t seg_snode,
				  index_t seg_nnodes,
				  index_t seg_idx)
        {
            dim3 grid_dims, block_dims;

            graph_datum.m_wl_array_in_seg[seg_idx].ResetAsync(stream.cuda_stream);

            KernelSizing(grid_dims, block_dims, seg_nnodes);

            kernel::RebuildWorklist
                    << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                    graph_datum.GetWorkSourceRangeDeviceObject(seg_snode,seg_nnodes),
                    graph_datum.m_wl_array_in_seg[seg_idx].DeviceObject(),
                    graph_datum.GetBufferDeviceObject(),
		            graph_datum.GetValueDeviceObject(),
                    graph_datum.activeNodesLabeling.dev_ptr,
                    graph_datum.activeNodesDegree.dev_ptr
		    );

            stream.Sync();
        }
        
        template<typename TAppInst, typename TGraphDatum>
        void RebuildArrayWorklist_zero(TAppInst &app_inst,
                                  TGraphDatum &graph_datum,
                                  const groute::Stream &stream,
                  index_t seg_snode,
                  index_t seg_nnodes,
                  index_t seg_idx)
        {
            dim3 grid_dims, block_dims;

            //graph_datum.m_wl_array_in_seg[seg_idx].ResetAsync(stream.cuda_stream);

            KernelSizing(grid_dims, block_dims, seg_nnodes);

            kernel::RebuildWorklist
                    << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                    graph_datum.GetWorkSourceRangeDeviceObject(seg_snode,seg_nnodes),
                    graph_datum.m_wl_array_in_seg[seg_idx].DeviceObject(),
                    graph_datum.GetBufferDeviceObject(),
                    graph_datum.GetValueDeviceObject(),
                    graph_datum.activeNodesLabeling.dev_ptr,
                    graph_datum.activeNodesDegree.dev_ptr
            );

            stream.Sync();
        }

        template<typename TAppInst, typename TGraphDatum>
        void RebuildArrayWorklist_compaction(TAppInst &app_inst,
                                  TGraphDatum &graph_datum,
                                  const groute::Stream &stream,
                  index_t seg_snode,
                  index_t seg_nnodes,
                  index_t seg_idx)
        {
            dim3 grid_dims, block_dims;

            //graph_datum.m_wl_array_in_seg[seg_idx].ResetAsync(stream.cuda_stream);

            KernelSizing(grid_dims, block_dims, seg_nnodes);

            kernel::RebuildWorklist_compaction
                    << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                    graph_datum.GetWorkSourceRangeDeviceObject(seg_snode,seg_nnodes),
                    graph_datum.m_wl_array_in_seg[seg_idx].DeviceObject(),
                    graph_datum.GetBufferDeviceObject(),
                    graph_datum.GetValueDeviceObject(),
                    graph_datum.activeNodesLabeling.dev_ptr,
                    graph_datum.activeNodesDegree.dev_ptr,
                    graph_datum.m_out_degree.dev_ptr
            );

            stream.Sync();
        }

        template<typename TAppInst, typename TGraphDatum>
        void RebuildBitmapWorklist(TAppInst &app_inst,
                                   TGraphDatum &graph_datum,
                                   const groute::Stream &stream)
        {
            dim3 grid_dims, block_dims;

            graph_datum.m_wl_bitmap_in.ResetAsync(stream.cuda_stream);

            KernelSizing(grid_dims, block_dims, graph_datum.nnodes);

            kernel::RebuildWorklist
                    << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                    graph_datum.GetWorkSourceRangeDeviceObject(),
                    graph_datum.m_wl_bitmap_in.DeviceObject(),
                    graph_datum.GetBufferDeviceObject());

            stream.Sync();
        }

        template<typename TGraphDatum, typename TBuffer>
        TBuffer GetPriorityThreshold(TGraphDatum &graph_datum,
                                     float cut_threshold,
                                     const groute::Stream &stream)
        {
            dim3 grid_dims, block_dims;
            size_t sample_size = min(PRIORITY_SAMPLE_SIZE, graph_datum.nnodes);

            for (int i = 0; i < sample_size; i++)
            {
                index_t node = std::rand() % graph_datum.nnodes;
                graph_datum.m_sampled_nodes.host_vec[i] = node;
            }

            graph_datum.m_sampled_nodes.H2DAsync(stream.cuda_stream);

            KernelSizing(grid_dims, block_dims, sample_size);
            kernel::Sample
                    << < grid_dims, block_dims, 0, stream.cuda_stream >> > (groute::dev::WorkSourceArray<index_t>(
                    graph_datum.m_sampled_nodes.dev_ptr, sample_size),
                    graph_datum.GetBufferDeviceObject(),
                    graph_datum.m_sampled_values.dev_ptr);
            graph_datum.m_sampled_values.D2HAsync(stream.cuda_stream);

            stream.Sync();

            thrust::device_ptr<TBuffer> p_sampled_values(graph_datum.m_sampled_values.dev_ptr);
            thrust::sort(p_sampled_values, p_sampled_values + sample_size);
            graph_datum.m_sampled_values.D2H();

            uint32_t cut_idx = sample_size * (1 - cut_threshold);

            return graph_datum.m_sampled_values.host_vec[cut_idx];
        }

        template<typename TAppInst,
                typename TCSRGraph,
                typename TGraphDatum>
        void RunAsyncPushDD(TAppInst &app_inst,
			    index_t seg_nnode,
			    index_t seg_snode,
			    uint64_t seg_sedge_csr,
                            TCSRGraph &csr_graph,
                            TGraphDatum &graph_datum,
                            EngineOptions &engine_options,
                            const groute::Stream &stream)
        {
            dim3 grid_dims, block_dims;
            uint32_t work_size = graph_datum.m_wl_array_in.GetCount(stream);

            KernelSizing(grid_dims, block_dims, work_size);

            switch (engine_options.GetLoadBalancing(common::MsgPassing::PUSH))
            {
                case LoadBalancing::NONE:
                    kernel::AsyncPushDD<LoadBalancing::NONE>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                            groute::dev::WorkSourceArray<index_t>(
                                    graph_datum.m_wl_array_in.GetDeviceDataPtr(),
                                    work_size),
                            graph_datum.m_wl_array_out_high.DeviceObject(),
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject());
                    break;
                case LoadBalancing::COARSE_GRAINED:
                    kernel::AsyncPushDD<LoadBalancing::COARSE_GRAINED>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                            groute::dev::WorkSourceArray<index_t>(
                                    graph_datum.m_wl_array_in.GetDeviceDataPtr(),
                                    work_size),
                            graph_datum.m_wl_array_out_high.DeviceObject(),
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject());
                    break;
                case LoadBalancing::FINE_GRAINED:
                    kernel::AsyncPushDD<LoadBalancing::FINE_GRAINED>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                            groute::dev::WorkSourceArray<index_t>(
                                    graph_datum.m_wl_array_in.GetDeviceDataPtr(),
                                    work_size),
                            graph_datum.m_wl_array_out_high.DeviceObject(),
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject());
                    break;
                case LoadBalancing::HYBRID:
                    kernel::AsyncPushDD<LoadBalancing::HYBRID>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                            groute::dev::WorkSourceArray<index_t>(
                                    graph_datum.m_wl_array_in.GetDeviceDataPtr(),
                                    work_size),
                            graph_datum.m_wl_array_out_high.DeviceObject(),
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject());
                    break;
            }

            graph_datum.m_wl_array_in.ResetAsync(stream);
            graph_datum.m_wl_array_in.Swap(graph_datum.m_wl_array_out_high);

            stream.Sync();
        }

 

        template<typename TAppInst,
                typename TCSRGraph,
                typename TGraphDatum>
        void RunSyncPushTDB(TAppInst &app_inst,
                index_t seg_snode,
                index_t seg_enode,
                uint64_t seg_sedge_csr,
		index_t seg_idx,
                bool zcflag,
                            TCSRGraph &csr_graph,
                            TGraphDatum &graph_datum,
                            EngineOptions &engine_options,
			    const groute::Stream &stream
                            )
        {
            dim3 grid_dims, block_dims;

            KernelSizing(grid_dims, block_dims, seg_enode-seg_snode);
            uint32_t work_size = graph_datum.m_wl_array_in_seg[seg_idx].GetCount(stream);
            switch (engine_options.GetLoadBalancing(common::MsgPassing::PUSH))
            {
                case LoadBalancing::NONE:
                    kernel::SyncPushTDB<LoadBalancing::NONE>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,seg_snode,seg_enode,seg_sedge_csr,zcflag,
                                 groute::dev::WorkSourceArray<index_t>(
                                        graph_datum.m_wl_array_in_seg[seg_idx].GetDeviceDataPtr(),
                                        work_size),
                            //graph_datum.GetWorkSourceRangeDeviceObject(seg_snode,seg_enode-seg_snode),
                            //graph_datum.m_wl_array_out_high.DeviceObject(),
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject(),
                            graph_datum.m_wl_bitmap_out_high.DeviceObject(),
                            graph_datum.m_wl_bitmap_in.DeviceObject());
                    break;
                case LoadBalancing::COARSE_GRAINED:
                    kernel::SyncPushTDB<LoadBalancing::COARSE_GRAINED>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,seg_snode,seg_enode,seg_sedge_csr,zcflag,
                                 groute::dev::WorkSourceArray<index_t>(
                                         graph_datum.m_wl_array_in_seg[seg_idx].GetDeviceDataPtr(),
                                         work_size),
                            //graph_datum.GetWorkSourceRangeDeviceObject(seg_snode,seg_enode-seg_snode),
                            //graph_datum.m_wl_array_out_high.DeviceObject(),
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject(),
                            graph_datum.m_wl_bitmap_out_high.DeviceObject(),
                            graph_datum.m_wl_bitmap_in.DeviceObject());
                    break;
                case LoadBalancing::FINE_GRAINED:
                    kernel::SyncPushTDB<LoadBalancing::FINE_GRAINED>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,seg_snode,seg_enode,seg_sedge_csr,zcflag,
                                 groute::dev::WorkSourceArray<index_t>(
                                         graph_datum.m_wl_array_in_seg[seg_idx].GetDeviceDataPtr(),
                                         work_size),
                            //graph_datum.GetWorkSourceRangeDeviceObject(seg_snode,seg_enode-seg_snode),
                            //graph_datum.m_wl_array_out_high.DeviceObject(),
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject(),
                            graph_datum.m_wl_bitmap_out_high.DeviceObject(),
                            graph_datum.m_wl_bitmap_in.DeviceObject());
                    break;
                case LoadBalancing::HYBRID:
                    kernel::SyncPushTDB<LoadBalancing::HYBRID>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,seg_snode,seg_enode,seg_sedge_csr,zcflag,
                                 groute::dev::WorkSourceArray<index_t>(
                                         graph_datum.m_wl_array_in_seg[seg_idx].GetDeviceDataPtr(),
                                         work_size),
                            //graph_datum.GetWorkSourceRangeDeviceObject(seg_snode,seg_enode-seg_snode),
                            //graph_datum.m_wl_array_out_high.DeviceObject(),
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject(),
                            graph_datum.m_wl_bitmap_out_high.DeviceObject(),
                            graph_datum.m_wl_bitmap_in.DeviceObject());
                    break;
                default:
                    assert(false);
            }

            //stream.Sync();
        }

	template<typename TAppInst,
                typename TCSRGraph,
                typename TGraphDatum>
        void RunSyncPushDDB(TAppInst &app_inst,
                index_t seg_snode,
                index_t seg_enode,
                uint64_t seg_sedge_csr,
		        index_t seg_idx,
                bool zcflag,
                            const TCSRGraph &csr_graph,
                            TGraphDatum &graph_datum,
                            EngineOptions &engine_options,
                            const groute::Stream &stream)
        {
            dim3 grid_dims, block_dims;
            KernelSizing(grid_dims, block_dims, seg_enode-seg_snode);
            uint32_t work_size = graph_datum.m_wl_array_in_seg[seg_idx].GetCount(stream);
            //printf("work_size:%d\n",work_size);
            if(zcflag)
                KernelSizing(grid_dims, block_dims, work_size);
            //KernelSizing(grid_dims, block_dims, work_size);
            switch (engine_options.GetLoadBalancing(common::MsgPassing::PUSH))
            {
                case LoadBalancing::NONE:
                    kernel::SyncPushDDB<LoadBalancing::NONE>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,seg_snode,seg_enode,seg_sedge_csr,zcflag,
                                 groute::dev::WorkSourceArray<index_t>(
                                        graph_datum.m_wl_array_in_seg[seg_idx].GetDeviceDataPtr(),
                                        work_size),
                            //graph_datum.GetWorkSourceRangeDeviceObject(seg_snode,seg_enode-seg_snode),
                            //graph_datum.m_wl_array_out_high.DeviceObject(),
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject(),
                            graph_datum.m_wl_bitmap_out_high.DeviceObject(),
                            graph_datum.m_wl_bitmap_in.DeviceObject());
                    break;
                case LoadBalancing::COARSE_GRAINED:
                    kernel::SyncPushDDB<LoadBalancing::COARSE_GRAINED>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,seg_snode,seg_enode,seg_sedge_csr,zcflag,
                                 groute::dev::WorkSourceArray<index_t>(
                                         graph_datum.m_wl_array_in_seg[seg_idx].GetDeviceDataPtr(),
                                         work_size),
                            //graph_datum.GetWorkSourceRangeDeviceObject(seg_snode,seg_enode-seg_snode),
                            //graph_datum.m_wl_array_out_high.DeviceObject(),
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject(),
                            graph_datum.m_wl_bitmap_out_high.DeviceObject(),
                            graph_datum.m_wl_bitmap_in.DeviceObject());
                    break;
                case LoadBalancing::FINE_GRAINED:
                    kernel::SyncPushDDB<LoadBalancing::FINE_GRAINED>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,seg_snode,seg_enode,seg_sedge_csr,zcflag,
                                 groute::dev::WorkSourceArray<index_t>(
                                         graph_datum.m_wl_array_in_seg[seg_idx].GetDeviceDataPtr(),
                                         work_size),
                            //graph_datum.GetWorkSourceRangeDeviceObject(seg_snode,seg_enode-seg_snode),
                            //graph_datum.m_wl_array_out_high.DeviceObject(),
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject(),
                            graph_datum.m_wl_bitmap_out_high.DeviceObject(),
                            graph_datum.m_wl_bitmap_in.DeviceObject());
                    break;
                case LoadBalancing::HYBRID:
                    kernel::SyncPushDDB<LoadBalancing::HYBRID>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,seg_snode,seg_enode,seg_sedge_csr,zcflag,
                                 groute::dev::WorkSourceArray<index_t>(
                                         graph_datum.m_wl_array_in_seg[seg_idx].GetDeviceDataPtr(),
                                         work_size),
                            //graph_datum.GetWorkSourceRangeDeviceObject(seg_snode,seg_enode-seg_snode),
                            //graph_datum.m_wl_array_out_high.DeviceObject(),
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject(),
                            graph_datum.m_wl_bitmap_out_high.DeviceObject(),
                            graph_datum.m_wl_bitmap_in.DeviceObject());
                    break;
                default:
                    assert(false);
            }


            //index_t activenum = graph_datum.m_wl_bitmap_out_high.GetPositiveCount(stream);
            //printf("segid:%d activenum:%d\n",seg_idx,activenum);
            //graph_datum.m_wl_bitmap_out_high.ResetAsync(stream);
            //stream.Sync();
        }

    template<typename TAppInst,
                typename TCSRGraph,
                typename TGraphDatum>
        void RunSyncPushDDB_COM(TAppInst &app_inst,
                                index_t active_count,
                                index_t seg_idx,
                                const TCSRGraph &csr_graph,
                                TGraphDatum &graph_datum,
                                EngineOptions &engine_options,
                                const groute::Stream &stream)
        {
            dim3 grid_dims, block_dims;
            KernelSizing(grid_dims, block_dims, active_count);
            uint32_t work_size = graph_datum.m_wl_array_in_seg[seg_idx].GetCount(stream);
            //printf("work_size:%d\n",work_size);
            switch (engine_options.GetLoadBalancing(common::MsgPassing::PUSH))
            {
                case LoadBalancing::NONE:
                    kernel::SyncPushDDB_COM<LoadBalancing::NONE>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                                 groute::dev::WorkSourceArray<index_t>(
                                        graph_datum.m_wl_array_in_seg[seg_idx].GetDeviceDataPtr(),
                                        work_size),
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject(),
                            graph_datum.m_wl_bitmap_out_high.DeviceObject(),
                            graph_datum.m_wl_bitmap_in.DeviceObject());
                    break;
                case LoadBalancing::COARSE_GRAINED:
                    kernel::SyncPushDDB_COM<LoadBalancing::COARSE_GRAINED>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                                 groute::dev::WorkSourceArray<index_t>(
                                         graph_datum.m_wl_array_in_seg[seg_idx].GetDeviceDataPtr(),
                                         work_size),
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject(),
                            graph_datum.m_wl_bitmap_out_high.DeviceObject(),
                            graph_datum.m_wl_bitmap_in.DeviceObject());
                    break;
                case LoadBalancing::FINE_GRAINED:
                    kernel::SyncPushDDB_COM<LoadBalancing::FINE_GRAINED>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                                 groute::dev::WorkSourceArray<index_t>(
                                         graph_datum.m_wl_array_in_seg[seg_idx].GetDeviceDataPtr(),
                                         work_size),
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject(),
                            graph_datum.m_wl_bitmap_out_high.DeviceObject(),
                            graph_datum.m_wl_bitmap_in.DeviceObject());
                    break;
                case LoadBalancing::HYBRID:
                    kernel::SyncPushDDB_COM<LoadBalancing::HYBRID>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                                 groute::dev::WorkSourceArray<index_t>(
                                         graph_datum.m_wl_array_in_seg[seg_idx].GetDeviceDataPtr(),
                                         work_size),
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject(),
                            graph_datum.m_wl_bitmap_out_high.DeviceObject(),
                            graph_datum.m_wl_bitmap_in.DeviceObject());
                    break;
                default:
                    assert(false);
            }


            //index_t activenum = graph_datum.m_wl_bitmap_out_high.GetPositiveCount(stream);
            //printf("segid:%d activenum:%d\n",seg_idx,activenum);
            //graph_datum.m_wl_bitmap_out_high.ResetAsync(stream);
            //stream.Sync();
        }
        /*forsep-graph+ */
        template<typename TAppInst,
                typename TCSRGraph,
                typename TGraphDatum>
        void RunSyncPushDD(TAppInst &app_inst,
			   index_t seg_snode,
			   index_t seg_enode,
			   uint64_t seg_sedge_csr,
			   bool zcflag,
                            TCSRGraph &csr_graph,
                            TGraphDatum &graph_datum,
                            EngineOptions &engine_options,
                            const groute::Stream &stream)
        {
            dim3 grid_dims, block_dims;
            uint32_t work_size = graph_datum.m_wl_array_in.GetCount(stream);

            KernelSizing(grid_dims, block_dims, work_size);

            switch (engine_options.GetLoadBalancing(common::MsgPassing::PUSH))
            {
                case LoadBalancing::NONE:
                    kernel::SyncPushDD<LoadBalancing::NONE>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,seg_snode,seg_enode,seg_sedge_csr,zcflag,
                            groute::dev::WorkSourceArray<index_t>(
                                        graph_datum.m_wl_array_in.GetDeviceDataPtr(),
                                        work_size),
                            graph_datum.m_wl_array_out_high.DeviceObject(),
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject());
                    break;
                case LoadBalancing::COARSE_GRAINED:
                    kernel::SyncPushDD<LoadBalancing::COARSE_GRAINED>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,seg_snode,seg_enode,seg_sedge_csr,zcflag,
                            groute::dev::WorkSourceArray<index_t>(
                                    graph_datum.m_wl_array_in.GetDeviceDataPtr(),
                                    work_size),
                            graph_datum.m_wl_array_out_high.DeviceObject(),
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject());
                    break;
                case LoadBalancing::FINE_GRAINED:
                    kernel::SyncPushDD<LoadBalancing::FINE_GRAINED>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,seg_snode,seg_enode,seg_sedge_csr,zcflag,
                            groute::dev::WorkSourceArray<index_t>(
                                    graph_datum.m_wl_array_in.GetDeviceDataPtr(),
                                    work_size),
                            graph_datum.m_wl_array_out_high.DeviceObject(),
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject());
                    break;
                case LoadBalancing::HYBRID:
                    kernel::SyncPushDD<LoadBalancing::HYBRID>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,seg_snode,seg_enode,seg_sedge_csr,zcflag,
                            groute::dev::WorkSourceArray<index_t>(
                                    graph_datum.m_wl_array_in.GetDeviceDataPtr(),
                                    work_size),
                            graph_datum.m_wl_array_out_high.DeviceObject(),
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject());
                    break;
            }

        //     graph_datum.m_wl_array_in.ResetAsync(stream);
        //     graph_datum.m_wl_array_in.Swap(graph_datum.m_wl_array_out_high);

            stream.Sync();
        }

        template<typename TAppInst,
                typename TBuffer,
                typename TCSRGraph,
                typename TGraphDatum>
        void RunAsyncPushDDPrio(TAppInst &app_inst,
				index_t seg_nnode,
			        index_t seg_snode,
			        uint64_t seg_sedge_csr,
                                TCSRGraph &csr_graph,
                                TBuffer &current_priority,
                                TGraphDatum &graph_datum,
                                EngineOptions &engine_options,
                                const groute::Stream &stream)
        {
            dim3 grid_dims, block_dims;
            uint32_t work_size = graph_datum.m_wl_array_in.GetCount(stream);

            KernelSizing(grid_dims, block_dims, work_size);

            switch (engine_options.GetLoadBalancing(common::MsgPassing::PUSH))
            {
                case LoadBalancing::NONE:
                    kernel::AsyncPushDDPrio<LoadBalancing::NONE>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                            groute::dev::WorkSourceArray<index_t>(
                                    graph_datum.m_wl_array_in.GetDeviceDataPtr(),
                                    work_size),
                            graph_datum.m_wl_array_out_low.DeviceObject(),
                            graph_datum.m_wl_array_out_high.DeviceObject(),
                            current_priority,
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject());
                    break;
                case LoadBalancing::COARSE_GRAINED:
                    kernel::AsyncPushDDPrio<LoadBalancing::COARSE_GRAINED>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                            groute::dev::WorkSourceArray<index_t>(
                                    graph_datum.m_wl_array_in.GetDeviceDataPtr(),
                                    work_size),
                            graph_datum.m_wl_array_out_low.DeviceObject(),
                            graph_datum.m_wl_array_out_high.DeviceObject(),
                            current_priority,
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject());
                    break;
                case LoadBalancing::FINE_GRAINED:
                    kernel::AsyncPushDDPrio<LoadBalancing::FINE_GRAINED>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                            groute::dev::WorkSourceArray<index_t>(
                                    graph_datum.m_wl_array_in.GetDeviceDataPtr(),
                                    work_size),
                            graph_datum.m_wl_array_out_low.DeviceObject(),
                            graph_datum.m_wl_array_out_high.DeviceObject(),
                            current_priority,
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject());
                    break;
                case LoadBalancing::HYBRID:
                    kernel::AsyncPushDDPrio<LoadBalancing::HYBRID>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                            groute::dev::WorkSourceArray<index_t>(
                                    graph_datum.m_wl_array_in.GetDeviceDataPtr(),
                                    work_size),
                            graph_datum.m_wl_array_out_low.DeviceObject(),
                            graph_datum.m_wl_array_out_high.DeviceObject(),
                            current_priority,
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject());
                    break;
            }

            if (FLAGS_trace)
            {
                printf("In: %u Out-low: %u Out-high: %u\n",
                       graph_datum.m_wl_array_in.GetCount(stream),
                       graph_datum.m_wl_array_out_low.GetCount(stream),
                       graph_datum.m_wl_array_out_high.GetCount(stream));
            }


            graph_datum.m_wl_array_in.ResetAsync(stream);
            if (graph_datum.m_wl_array_out_high.GetCount(stream) > 0)
            {
                graph_datum.m_wl_array_in.Swap(graph_datum.m_wl_array_out_high);
            }
            else // use out of high priority queue
            {
                current_priority += engine_options.GetPriorityThreshold();
                graph_datum.m_wl_array_in.Swap(graph_datum.m_wl_array_out_low);
            }

            stream.Sync();
        }

        template<typename TAppInst,
                typename TCSRGraph,
                typename TGraphDatum,
                typename TValue,
                typename TBuffer,
                typename TWeight>
        void RunAsyncPushDDFused(TAppInst &app_inst,
				 index_t seg_nnode,
				 index_t seg_snode,
				 uint64_t seg_sedge_csr,
                                 TCSRGraph &csr_graph,
                                 TGraphDatum &graph_datum,
                                 EngineOptions &engine_options,
                                 cudaDeviceProp &dev_props,
                                 const groute::Stream &stream)
        {
            dim3 grid_dims, block_dims;
            int occupancy_per_MP = 0;

            switch (engine_options.GetLoadBalancing(common::MsgPassing::PUSH))
            {
                case LoadBalancing::NONE:
                    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy_per_MP,
                                                                  kernel::AsyncPushDDFused<LoadBalancing::NONE, TAppInst,
                                                                          TCSRGraph, groute::graphs::dev::GraphDatum, TValue, TBuffer, TWeight>,
                                                                  FLAGS_block_size, 0);
                    break;
                case LoadBalancing::COARSE_GRAINED:
                    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy_per_MP,
                                                                  kernel::AsyncPushDDFused<LoadBalancing::COARSE_GRAINED, TAppInst,
                                                                          TCSRGraph, groute::graphs::dev::GraphDatum, TValue, TBuffer, TWeight>,
                                                                  FLAGS_block_size, 0);
                    break;
                case LoadBalancing::FINE_GRAINED:
                    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy_per_MP,
                                                                  kernel::AsyncPushDDFused<LoadBalancing::FINE_GRAINED, TAppInst,
                                                                          TCSRGraph, groute::graphs::dev::GraphDatum, TValue, TBuffer, TWeight>,
                                                                  FLAGS_block_size, 0);
                    break;
                case LoadBalancing::HYBRID:
                    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy_per_MP,
                                                                  kernel::AsyncPushDDFused<LoadBalancing::HYBRID, TAppInst,
                                                                          TCSRGraph, groute::graphs::dev::GraphDatum, TValue, TBuffer, TWeight>,
                                                                  FLAGS_block_size, 0);
                    break;
            }

            size_t fused_work_blocks = dev_props.multiProcessorCount * occupancy_per_MP;

            grid_dims.x = fused_work_blocks;
            block_dims.x = FLAGS_block_size;

            cub::GridBarrierLifetime barrier;

            barrier.Setup(fused_work_blocks);

            switch (engine_options.GetLoadBalancing(common::MsgPassing::PUSH))
            {
                case LoadBalancing::NONE:
                    kernel::AsyncPushDDFused<LoadBalancing::NONE>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                            graph_datum.m_wl_array_in.DeviceObject(),
                            graph_datum.m_wl_array_out_high.DeviceObject(),
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject(),
                            barrier);
                    break;
                case LoadBalancing::COARSE_GRAINED:
                    kernel::AsyncPushDDFused<LoadBalancing::COARSE_GRAINED>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                            graph_datum.m_wl_array_in.DeviceObject(),
                            graph_datum.m_wl_array_out_high.DeviceObject(),
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject(),
                            barrier);
                    break;
                case LoadBalancing::FINE_GRAINED:
                    kernel::AsyncPushDDFused<LoadBalancing::FINE_GRAINED>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                            graph_datum.m_wl_array_in.DeviceObject(),
                            graph_datum.m_wl_array_out_high.DeviceObject(),
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject(),
                            barrier);
                    break;
                case LoadBalancing::HYBRID:
                    kernel::AsyncPushDDFused<LoadBalancing::HYBRID>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                            graph_datum.m_wl_array_in.DeviceObject(),
                            graph_datum.m_wl_array_out_high.DeviceObject(),
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject(),
                            barrier);
                    break;
                default:
                    assert(false);
            }
            stream.Sync();
        }

        template<typename TAppInst,
                typename TCSRGraph,
                typename TGraphDatum,
                typename TValue,
                typename TBuffer,
                typename TWeight>
        void RunAsyncPushDDFusedPrio(TAppInst &app_inst,
				     index_t seg_nnode,
				     index_t seg_snode,
				     uint64_t seg_sedge_csr,
                                     TCSRGraph &csr_graph,
                                     TGraphDatum &graph_datum,
                                     EngineOptions &engine_options,
                                     cudaDeviceProp &dev_props,
                                     const groute::Stream &stream)
        {
            dim3 grid_dims, block_dims;
            int occupancy_per_MP = 0;
            TBuffer init_prio = engine_options.GetPriorityThreshold();

            switch (engine_options.GetLoadBalancing(common::MsgPassing::PUSH))
            {
                case LoadBalancing::NONE:
                    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy_per_MP,
                                                                  kernel::AsyncPushDDFusedPrio<LoadBalancing::NONE, TAppInst, TCSRGraph,
                                                                          groute::graphs::dev::GraphDatum, TValue, TBuffer, TWeight>,
                                                                  FLAGS_block_size, 0);
                    break;
                case LoadBalancing::COARSE_GRAINED:
                    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy_per_MP,
                                                                  kernel::AsyncPushDDFusedPrio<LoadBalancing::COARSE_GRAINED, TAppInst, TCSRGraph,
                                                                          groute::graphs::dev::GraphDatum, TValue, TBuffer, TWeight>,
                                                                  FLAGS_block_size, 0);
                    break;
                case LoadBalancing::FINE_GRAINED:
                    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy_per_MP,
                                                                  kernel::AsyncPushDDFusedPrio<LoadBalancing::FINE_GRAINED, TAppInst, TCSRGraph,
                                                                          groute::graphs::dev::GraphDatum, TValue, TBuffer, TWeight>,
                                                                  FLAGS_block_size, 0);
                    break;
                case LoadBalancing::HYBRID:
                    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy_per_MP,
                                                                  kernel::AsyncPushDDFusedPrio<LoadBalancing::HYBRID, TAppInst, TCSRGraph,
                                                                          groute::graphs::dev::GraphDatum, TValue, TBuffer, TWeight>,
                                                                  FLAGS_block_size, 0);
                    break;
            }

            size_t fused_work_blocks = dev_props.multiProcessorCount * occupancy_per_MP;

            grid_dims.x = fused_work_blocks;
            block_dims.x = FLAGS_block_size;

            cub::GridBarrierLifetime barrier;

            barrier.Setup(fused_work_blocks);

            switch (engine_options.GetLoadBalancing(common::MsgPassing::PUSH))
            {
                case LoadBalancing::NONE:
                    kernel::AsyncPushDDFusedPrio<LoadBalancing::NONE>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                            graph_datum.m_wl_array_in.DeviceObject(),
                            graph_datum.m_wl_array_out_low.DeviceObject(),
                            graph_datum.m_wl_array_out_high.DeviceObject(),
                            init_prio,
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject(),
                            barrier);
                    break;
                case LoadBalancing::COARSE_GRAINED:
                    kernel::AsyncPushDDFusedPrio<LoadBalancing::COARSE_GRAINED>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                            graph_datum.m_wl_array_in.DeviceObject(),
                            graph_datum.m_wl_array_out_low.DeviceObject(),
                            graph_datum.m_wl_array_out_high.DeviceObject(),
                            init_prio,
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject(),
                            barrier);
                    break;
                case LoadBalancing::FINE_GRAINED:
                    kernel::AsyncPushDDFusedPrio<LoadBalancing::FINE_GRAINED>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                            graph_datum.m_wl_array_in.DeviceObject(),
                            graph_datum.m_wl_array_out_low.DeviceObject(),
                            graph_datum.m_wl_array_out_high.DeviceObject(),
                            init_prio,
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject(),
                            barrier);
                    break;
                case LoadBalancing::HYBRID:
                    kernel::AsyncPushDDFusedPrio<LoadBalancing::HYBRID>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                            graph_datum.m_wl_array_in.DeviceObject(),
                            graph_datum.m_wl_array_out_low.DeviceObject(),
                            graph_datum.m_wl_array_out_high.DeviceObject(),
                            init_prio,
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject(),
                            barrier);
                    break;
            }
            stream.Sync();
        }

        template<typename TAppInst,
                typename TCSRGraph,
                typename TGraphDatum>
        void RunAsyncPushTD(TAppInst &app_inst,
			    index_t seg_nnode,
			    index_t seg_snode,
			    uint64_t seg_sedge_csr,
                            TCSRGraph &csr_graph,
                            TGraphDatum &graph_datum,
                            EngineOptions &engine_options,
                            const groute::Stream &stream)
        {
            dim3 grid_dims, block_dims;

            KernelSizing(grid_dims, block_dims, graph_datum.nnodes);

            switch (engine_options.GetLoadBalancing(common::MsgPassing::PUSH))
            {
                case LoadBalancing::NONE:
                    kernel::AsyncPushTD<LoadBalancing::NONE>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                            graph_datum.GetWorkSourceRangeDeviceObject(),
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject());
                    break;
                case LoadBalancing::COARSE_GRAINED:
                    kernel::AsyncPushTD<LoadBalancing::COARSE_GRAINED>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                            graph_datum.GetWorkSourceRangeDeviceObject(),
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject());
                    break;
                case LoadBalancing::FINE_GRAINED:
                    kernel::AsyncPushTD<LoadBalancing::FINE_GRAINED>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                            graph_datum.GetWorkSourceRangeDeviceObject(),
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject());
                    break;
                case LoadBalancing::HYBRID:
                    kernel::AsyncPushTD<LoadBalancing::HYBRID>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                            graph_datum.GetWorkSourceRangeDeviceObject(),
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject());
                    break;
                default:
                    assert(false);
            }

            stream.Sync();
        }


        template<typename TAppInst,
                typename TCSRGraph,
                typename TGraphDatum>
        void RunSyncPushTD(TAppInst &app_inst,
			   index_t seg_nnode,
			   index_t seg_snode,
			   uint64_t seg_sedge_csr,
                            TCSRGraph &csr_graph,
                            TGraphDatum &graph_datum,
                            EngineOptions &engine_options,
                            const groute::Stream &stream)
        {
            dim3 grid_dims, block_dims;

            KernelSizing(grid_dims, block_dims, graph_datum.nnodes);

            switch (engine_options.GetLoadBalancing(common::MsgPassing::PUSH))
            {
                case LoadBalancing::NONE:
                    kernel::SyncPushTD<LoadBalancing::NONE>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                            graph_datum.GetWorkSourceRangeDeviceObject(seg_snode,seg_nnode),
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject());
                    break;
                case LoadBalancing::COARSE_GRAINED:
                    kernel::SyncPushTD<LoadBalancing::COARSE_GRAINED>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                            graph_datum.GetWorkSourceRangeDeviceObject(seg_snode,seg_nnode),
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject());
                    break;
                case LoadBalancing::FINE_GRAINED:
                    kernel::SyncPushTD<LoadBalancing::FINE_GRAINED>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                            graph_datum.GetWorkSourceRangeDeviceObject(seg_snode,seg_nnode),
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject());
                    break;
                case LoadBalancing::HYBRID:
                    kernel::SyncPushTD<LoadBalancing::HYBRID>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                            graph_datum.GetWorkSourceRangeDeviceObject(seg_snode,seg_nnode),
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject());
                    break;
                default:
                    assert(false);
            }

            stream.Sync();
        }

        template<typename TAppInst,
                typename TCSRGraph,
                typename TGraphDatum,
                typename TBuffer>
        void RunAsyncPushTDPrio(TAppInst &app_inst,
				index_t seg_nnode,
				index_t seg_snode,
				uint64_t seg_sedge_csr,
                                TCSRGraph &csr_graph,
                                TGraphDatum &graph_datum,
                                EngineOptions &engine_options,
                                TBuffer priority,
                                const groute::Stream &stream)
        {
            dim3 grid_dims, block_dims;

            KernelSizing(grid_dims, block_dims, graph_datum.nnodes);

            switch (engine_options.GetLoadBalancing(common::MsgPassing::PUSH))
            {
                case LoadBalancing::NONE:
                    kernel::AsyncPushTDPrio<LoadBalancing::NONE>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                            graph_datum.GetWorkSourceRangeDeviceObject(seg_snode, seg_nnode),
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject(),
                            priority);
                    break;
                case LoadBalancing::COARSE_GRAINED:
                    kernel::AsyncPushTDPrio<LoadBalancing::COARSE_GRAINED>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                            graph_datum.GetWorkSourceRangeDeviceObject(seg_snode, seg_nnode),
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject(),
                            priority);
                    break;
                case LoadBalancing::FINE_GRAINED:
                    kernel::AsyncPushTDPrio<LoadBalancing::FINE_GRAINED>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                            graph_datum.GetWorkSourceRangeDeviceObject(seg_snode, seg_nnode),
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject(),
                            priority);
                    break;
                case LoadBalancing::HYBRID:
                    kernel::AsyncPushTDPrio<LoadBalancing::HYBRID>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                            graph_datum.GetWorkSourceRangeDeviceObject(seg_snode, seg_nnode),
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject(),
                            priority);
                    break;
                default:
                    assert(false);
            }

            stream.Sync();
        }

        template<typename TAppInst,
                typename TCSRGraph,
                typename TGraphDatum,
                typename TValue,
                typename TBuffer,
                typename TWeight>
        void RunAsyncPushTDFused(TAppInst &app_inst,
				 index_t seg_nnode,
				 index_t seg_snode,
				 uint64_t seg_sedge_csr,
                                 TCSRGraph &csr_graph,
                                 TGraphDatum &graph_datum,
                                 EngineOptions &engine_options,
                                 cudaDeviceProp &dev_props,
                                 const groute::Stream &stream)
        {
            dim3 grid_dims, block_dims;
            int occupancy_per_MP = 0;

            KernelSizing(grid_dims, block_dims, graph_datum.nnodes);
            graph_datum.m_active_nodes.set_val_H2DAsync(graph_datum.nnodes, stream.cuda_stream);

            switch (engine_options.GetLoadBalancing(common::MsgPassing::PUSH))
            {
                case LoadBalancing::NONE:
                    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy_per_MP,
                                                                  kernel::AsyncPushTDFused<LoadBalancing::NONE, TAppInst, groute::dev::WorkSourceRange<index_t>,
                                                                          TCSRGraph, groute::graphs::dev::GraphDatum, TValue, TBuffer, TWeight>,
                                                                  FLAGS_block_size, 0);
                    break;
                case LoadBalancing::COARSE_GRAINED:
                    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy_per_MP,
                                                                  kernel::AsyncPushTDFused<LoadBalancing::COARSE_GRAINED, TAppInst, groute::dev::WorkSourceRange<index_t>,
                                                                          TCSRGraph, groute::graphs::dev::GraphDatum, TValue, TBuffer, TWeight>,
                                                                  FLAGS_block_size, 0);
                    break;
                case LoadBalancing::FINE_GRAINED:
                    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy_per_MP,
                                                                  kernel::AsyncPushTDFused<LoadBalancing::FINE_GRAINED, TAppInst, groute::dev::WorkSourceRange<index_t>,
                                                                          TCSRGraph, groute::graphs::dev::GraphDatum, TValue, TBuffer, TWeight>,
                                                                  FLAGS_block_size, 0);
                    break;
                case LoadBalancing::HYBRID:
                    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy_per_MP,
                                                                  kernel::AsyncPushTDFused<LoadBalancing::HYBRID, TAppInst, groute::dev::WorkSourceRange<index_t>,
                                                                          TCSRGraph, groute::graphs::dev::GraphDatum, TValue, TBuffer, TWeight>,
                                                                  FLAGS_block_size, 0);
                    break;
            }

            size_t fused_work_blocks = dev_props.multiProcessorCount * occupancy_per_MP;

            grid_dims.x = fused_work_blocks;
            block_dims.x = FLAGS_block_size;

            cub::GridBarrierLifetime barrier;

            barrier.Setup(fused_work_blocks);

            switch (engine_options.GetLoadBalancing(common::MsgPassing::PUSH))
            {
                case LoadBalancing::NONE:
                    kernel::AsyncPushTDFused<LoadBalancing::NONE>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                            graph_datum.GetWorkSourceRangeDeviceObject(seg_snode,seg_nnode),
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject(),
                            barrier,
                            graph_datum.m_active_nodes.dev_ptr);
                    break;
                case LoadBalancing::COARSE_GRAINED:
                    kernel::AsyncPushTDFused<LoadBalancing::COARSE_GRAINED>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                            graph_datum.GetWorkSourceRangeDeviceObject(seg_snode,seg_nnode),
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject(),
                            barrier,
                            graph_datum.m_active_nodes.dev_ptr);
                    break;
                case LoadBalancing::FINE_GRAINED:
                    kernel::AsyncPushTDFused<LoadBalancing::FINE_GRAINED>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                            graph_datum.GetWorkSourceRangeDeviceObject(seg_snode,seg_nnode),
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject(),
                            barrier,
                            graph_datum.m_active_nodes.dev_ptr);
                    break;
                case LoadBalancing::HYBRID:
                    kernel::AsyncPushTDFused<LoadBalancing::HYBRID>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                            graph_datum.GetWorkSourceRangeDeviceObject(seg_snode,seg_nnode),
                            csr_graph,
                            graph_datum.GetValueDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject(),
                            barrier,
                            graph_datum.m_active_nodes.dev_ptr);
                    break;
                default:
                    assert(false);
            }

            stream.Sync();
        }


        template<typename TAppInst,
                typename TCSCGraph,
                typename TGraphDatum>
        void RunSyncPullTD(TAppInst &app_inst,
			   index_t seg_nnode,
			   index_t seg_snode,
			   uint64_t seg_sedge_csc,
                           TCSCGraph csc_graph,
                           TGraphDatum &graph_datum,
                           EngineOptions &engine_options,
                           const groute::Stream &stream)
        {
            dim3 grid_dims, block_dims;

            KernelSizing(grid_dims, block_dims, graph_datum.nnodes);

            graph_datum.m_wl_middle.ResetAsync(stream);

            kernel::common::CombineValueBuffer
                    << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                    graph_datum.GetWorkSourceRangeDeviceObject(seg_snode,seg_nnode),
                    graph_datum.m_wl_middle.DeviceObject(),
                    graph_datum.GetValueDeviceObject(),
                    graph_datum.GetBufferDeviceObject(),
                    graph_datum.GetBufferTmpDeviceObject());

            groute::dev::WorkSourceArray<index_t> work_source(graph_datum.m_wl_middle.GetDeviceDataPtr(),
                                                              graph_datum.m_wl_middle.GetCount(stream));

            KernelSizing(grid_dims, block_dims, work_source.get_size());

            switch (engine_options.GetLoadBalancing(common::MsgPassing::PULL))
            {
                case LoadBalancing::NONE:

                    kernel::SyncPullTD<LoadBalancing::NONE>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,seg_snode,seg_sedge_csc,
                            work_source,
                            csc_graph,
                            graph_datum.GetBufferTmpDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject());
                    break;
                case LoadBalancing::COARSE_GRAINED:
                    kernel::SyncPullTD<LoadBalancing::COARSE_GRAINED>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,seg_snode,seg_sedge_csc,
                            work_source,
                            csc_graph,
                            graph_datum.GetBufferTmpDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject());
                    break;
                case LoadBalancing::FINE_GRAINED:
                    kernel::SyncPullTD<LoadBalancing::FINE_GRAINED>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,seg_snode,seg_sedge_csc,
                            work_source,
                            csc_graph,
                            graph_datum.GetBufferTmpDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject());
                    break;
                case LoadBalancing::HYBRID:
                    kernel::SyncPullTD<LoadBalancing::HYBRID>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,seg_snode,seg_sedge_csc,
                            work_source,
                            csc_graph,
                            graph_datum.GetBufferTmpDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject());
                    break;
                default:
                    assert(false);
            }
            stream.Sync();
        }

        template<typename TAppInst,
                typename TCSCGraph,
                typename TGraphDatum>
        void RunAsyncPullDD(TAppInst &app_inst,
			    index_t seg_nnode,
			    index_t seg_snode,
			    uint64_t seg_sedge_csc,
                            TCSCGraph csc_graph,
                            TGraphDatum &graph_datum,
                            EngineOptions &engine_options,
                            const groute::Stream &stream)
        {
            dim3 grid_dims, block_dims;

            KernelSizing(grid_dims, block_dims, graph_datum.nnodes);

            graph_datum.m_wl_middle.ResetAsync(stream);
            // we can try async pull dd, we split the procedure like above to prevent clear delta before use it
            kernel::common::CombineValueBuffer
                    << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                    graph_datum.GetWorkSourceRangeDeviceObject(seg_snode,seg_nnode),
                    graph_datum.m_wl_middle.DeviceObject(),
                    graph_datum.GetValueDeviceObject(),
                    graph_datum.GetBufferDeviceObject(),
                    graph_datum.GetBufferDeviceObject());

            groute::dev::WorkSourceArray<index_t> work_source(graph_datum.m_wl_middle.GetDeviceDataPtr(),
                                                              graph_datum.m_wl_middle.GetCount(stream));

            KernelSizing(grid_dims, block_dims, work_source.get_size());

            switch (engine_options.GetLoadBalancing(common::MsgPassing::PULL))
            {
                case LoadBalancing::NONE:
                    kernel::AsyncPullDD<LoadBalancing::NONE>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                            work_source,
                            graph_datum.m_wl_bitmap_in.DeviceObject(),
                            graph_datum.m_wl_bitmap_out_high.DeviceObject(),
                            csc_graph,
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetCSCEdgeWeightDeviceObject());
                    break;
                case LoadBalancing::COARSE_GRAINED:
                    kernel::AsyncPullDD<LoadBalancing::COARSE_GRAINED>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                            work_source,
                            graph_datum.m_wl_bitmap_in.DeviceObject(),
                            graph_datum.m_wl_bitmap_out_high.DeviceObject(),
                            csc_graph,
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetCSCEdgeWeightDeviceObject());
                    break;
                case LoadBalancing::FINE_GRAINED:
                    kernel::AsyncPullDD<LoadBalancing::FINE_GRAINED>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                            work_source,
                            graph_datum.m_wl_bitmap_in.DeviceObject(),
                            graph_datum.m_wl_bitmap_out_high.DeviceObject(),
                            csc_graph,
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetCSCEdgeWeightDeviceObject());
                    break;
                case LoadBalancing::HYBRID:
                    kernel::AsyncPullDD<LoadBalancing::HYBRID>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                            work_source,
                            graph_datum.m_wl_bitmap_in.DeviceObject(),
                            graph_datum.m_wl_bitmap_out_high.DeviceObject(),
                            csc_graph,
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetCSCEdgeWeightDeviceObject());
                    break;
                default:
                    assert(false);
            }

            graph_datum.m_wl_bitmap_in.ResetAsync(stream);
            stream.Sync();

            graph_datum.m_wl_bitmap_in.Swap(graph_datum.m_wl_bitmap_out_high);
        }

        template<typename TAppInst,
                typename TBuffer,
                typename TCSCGraph,
                typename TGraphDatum>
        void RunAsyncPullDDPrio(TAppInst &app_inst,
				index_t seg_nnode,
				index_t seg_snode,
				uint64_t seg_sedge_csc,
                                TCSCGraph csc_graph,
                                TBuffer &current_priority,
                                TGraphDatum &graph_datum,
                                EngineOptions &engine_options,
                                const groute::Stream &stream)
        {
            dim3 grid_dims, block_dims;

            KernelSizing(grid_dims, block_dims, graph_datum.nnodes);

            graph_datum.m_wl_middle.ResetAsync(stream);
            // we can try async pull dd, we split the procedure like above to prevent clear delta before use it
            kernel::common::CombineValueBuffer
                    << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                    graph_datum.GetWorkSourceRangeDeviceObject(seg_snode,seg_nnode),
                    graph_datum.m_wl_middle.DeviceObject(),
                    graph_datum.GetValueDeviceObject(),
                    graph_datum.GetBufferDeviceObject(),
                    graph_datum.GetBufferDeviceObject());


            groute::dev::WorkSourceArray<index_t> work_source(graph_datum.m_wl_middle.GetDeviceDataPtr(),
                                                              graph_datum.m_wl_middle.GetCount(stream));

            KernelSizing(grid_dims, block_dims, work_source.get_size());

            switch (engine_options.GetLoadBalancing(common::MsgPassing::PULL))
            {
                case LoadBalancing::NONE:
                    kernel::AsyncPullDDPrio<LoadBalancing::NONE>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                            work_source,
                            graph_datum.m_wl_bitmap_in.DeviceObject(),
                            graph_datum.m_wl_bitmap_out_low.DeviceObject(),
                            graph_datum.m_wl_bitmap_out_high.DeviceObject(),
                            current_priority,
                            csc_graph,
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetCSCEdgeWeightDeviceObject());
                    break;
                case LoadBalancing::COARSE_GRAINED:
                    kernel::AsyncPullDDPrio<LoadBalancing::COARSE_GRAINED>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                            work_source,
                            graph_datum.m_wl_bitmap_in.DeviceObject(),
                            graph_datum.m_wl_bitmap_out_low.DeviceObject(),
                            graph_datum.m_wl_bitmap_out_high.DeviceObject(),
                            current_priority,
                            csc_graph,
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetCSCEdgeWeightDeviceObject());
                    break;
                case LoadBalancing::FINE_GRAINED:
                    kernel::AsyncPullDDPrio<LoadBalancing::FINE_GRAINED>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                            work_source,
                            graph_datum.m_wl_bitmap_in.DeviceObject(),
                            graph_datum.m_wl_bitmap_out_low.DeviceObject(),
                            graph_datum.m_wl_bitmap_out_high.DeviceObject(),
                            current_priority,
                            csc_graph,
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetCSCEdgeWeightDeviceObject());
                    break;
                case LoadBalancing::HYBRID:
                    kernel::AsyncPullDDPrio<LoadBalancing::HYBRID>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                            work_source,
                            graph_datum.m_wl_bitmap_in.DeviceObject(),
                            graph_datum.m_wl_bitmap_out_low.DeviceObject(),
                            graph_datum.m_wl_bitmap_out_high.DeviceObject(),
                            current_priority,
                            csc_graph,
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetCSCEdgeWeightDeviceObject());
                    break;
                default:
                    assert(false);
            }

            if (FLAGS_trace)
            {
                printf("In: %u Out-low: %u Out-high: %u\n",
                       graph_datum.m_wl_bitmap_in.GetPositiveCount(stream),
                       graph_datum.m_wl_bitmap_out_low.GetPositiveCount(stream),
                       graph_datum.m_wl_bitmap_out_high.GetPositiveCount(stream));
            }

            graph_datum.m_wl_bitmap_in.ResetAsync(stream);

            if (graph_datum.m_wl_bitmap_out_high.GetPositiveCount(stream) > 0)
            {
                graph_datum.m_wl_bitmap_in.Swap(graph_datum.m_wl_bitmap_out_high);
            }
            else // use out of high priority queue
            {
                current_priority += engine_options.GetPriorityThreshold();
                graph_datum.m_wl_bitmap_in.Swap(graph_datum.m_wl_bitmap_out_low);
            }

            stream.Sync();
        }

        template<typename TAppInst,
                typename TCSCGraph,
                typename TGraphDatum>
        void RunSyncPullDD(TAppInst &app_inst,
			   index_t seg_snode,
			   index_t seg_enode,
			   uint64_t seg_sedge_csc,
			   bool zcflag,
                           TCSCGraph csc_graph,
                           TGraphDatum &graph_datum,
                           EngineOptions &engine_options,
                           const groute::Stream &stream)
        {
            dim3 grid_dims, block_dims;

            KernelSizing(grid_dims, block_dims, seg_enode - seg_snode);
            graph_datum.m_wl_middle.ResetAsync(stream);
            // we can try async pull dd, we split the procedure like above to prevent clear delta before use it
            kernel::common::CombineValueBuffer
                    << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                    graph_datum.GetWorkSourceRangeDeviceObject(seg_snode,seg_enode - seg_snode),
                    graph_datum.m_wl_middle.DeviceObject(),
                    graph_datum.GetValueDeviceObject(),
                    graph_datum.GetBufferDeviceObject(),
                    graph_datum.GetBufferTmpDeviceObject());


            groute::dev::WorkSourceArray<index_t> work_source(graph_datum.m_wl_middle.GetDeviceDataPtr(),
                                                              graph_datum.m_wl_middle.GetCount(stream));

            KernelSizing(grid_dims, block_dims, work_source.get_size());

            switch (engine_options.GetLoadBalancing(common::MsgPassing::PULL))
            {
                case LoadBalancing::NONE:
                    kernel::SyncPullDD<LoadBalancing::NONE>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,seg_snode,seg_enode,seg_sedge_csc,zcflag,
                            work_source,
                            graph_datum.m_wl_bitmap_in.DeviceObject(),
                            graph_datum.m_wl_bitmap_out_high.DeviceObject(),
                            csc_graph,
                            graph_datum.GetBufferTmpDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetCSCEdgeWeightDeviceObject());
                    break;
                case LoadBalancing::COARSE_GRAINED:
                    kernel::SyncPullDD<LoadBalancing::COARSE_GRAINED>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,seg_snode,seg_enode,seg_sedge_csc,zcflag,
                            work_source,
                            graph_datum.m_wl_bitmap_in.DeviceObject(),
                            graph_datum.m_wl_bitmap_out_high.DeviceObject(),
                            csc_graph,
                            graph_datum.GetBufferTmpDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetCSCEdgeWeightDeviceObject());
                    break;
                case LoadBalancing::FINE_GRAINED:
                    kernel::SyncPullDD<LoadBalancing::FINE_GRAINED>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,seg_snode,seg_enode,seg_sedge_csc,zcflag,
                            work_source,
                            graph_datum.m_wl_bitmap_in.DeviceObject(),
                            graph_datum.m_wl_bitmap_out_high.DeviceObject(),
                            csc_graph,
                            graph_datum.GetBufferTmpDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetCSCEdgeWeightDeviceObject());
                    break;
                case LoadBalancing::HYBRID:
                    kernel::SyncPullDD<LoadBalancing::HYBRID>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,seg_snode,seg_enode,seg_sedge_csc,zcflag,
                            work_source,
                            graph_datum.m_wl_bitmap_in.DeviceObject(),
                            graph_datum.m_wl_bitmap_out_high.DeviceObject(),
                            csc_graph,
                            graph_datum.GetBufferTmpDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetCSCEdgeWeightDeviceObject());
                    break;
                default:
                    assert(false);
            }

            
	    
	      stream.Sync();
        }
        template<typename TAppInst,
                typename TCSCGraph,
                typename TGraphDatum>
        void RunSyncPullTDB(TAppInst &app_inst,
			   index_t seg_nnode,
			   index_t seg_snode,
			   uint64_t seg_sedge_csc,
			    bool zcflag,
                           TCSCGraph csc_graph,
                           TGraphDatum &graph_datum,
                           EngineOptions &engine_options,
                           const groute::Stream &stream)
        {
            dim3 grid_dims, block_dims;

            KernelSizing(grid_dims, block_dims, graph_datum.nnodes);

            graph_datum.m_wl_middle.ResetAsync(stream);

            kernel::common::CombineValueBuffer
                    << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,
                    graph_datum.GetWorkSourceRangeDeviceObject(seg_snode,seg_nnode),
                    graph_datum.m_wl_middle.DeviceObject(),
                    graph_datum.GetValueDeviceObject(),
                    graph_datum.GetBufferDeviceObject(),
                    graph_datum.GetBufferTmpDeviceObject());

            groute::dev::WorkSourceArray<index_t> work_source(graph_datum.m_wl_middle.GetDeviceDataPtr(),
                                                              graph_datum.m_wl_middle.GetCount(stream));

            KernelSizing(grid_dims, block_dims, work_source.get_size());

            switch (engine_options.GetLoadBalancing(common::MsgPassing::PULL))
            {
                case LoadBalancing::NONE:

                    kernel::SyncPullTDB<LoadBalancing::NONE>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,seg_snode,seg_sedge_csc,zcflag,
                            work_source,
                            csc_graph,
                            graph_datum.GetBufferTmpDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject(),
			    graph_datum.m_wl_bitmap_out_high.DeviceObject(),
                            graph_datum.m_wl_bitmap_in.DeviceObject()
			    );
                    break;
                case LoadBalancing::COARSE_GRAINED:
                    kernel::SyncPullTDB<LoadBalancing::COARSE_GRAINED>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,seg_snode,seg_sedge_csc,zcflag,
                            work_source,
                            csc_graph,
                            graph_datum.GetBufferTmpDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject(),
			    graph_datum.m_wl_bitmap_out_high.DeviceObject(),
                            graph_datum.m_wl_bitmap_in.DeviceObject()
			    );
                    break;
                case LoadBalancing::FINE_GRAINED:
                    kernel::SyncPullTDB<LoadBalancing::FINE_GRAINED>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,seg_snode,seg_sedge_csc,zcflag,
                            work_source,
                            csc_graph,
                            graph_datum.GetBufferTmpDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject(),
			    graph_datum.m_wl_bitmap_out_high.DeviceObject(),
                            graph_datum.m_wl_bitmap_in.DeviceObject()
			    );
                    break;
                case LoadBalancing::HYBRID:
                    kernel::SyncPullTDB<LoadBalancing::HYBRID>
                            << < grid_dims, block_dims, 0, stream.cuda_stream >> > (app_inst,seg_snode,seg_sedge_csc,zcflag,
                            work_source,
                            csc_graph,
                            graph_datum.GetBufferTmpDeviceObject(),
                            graph_datum.GetBufferDeviceObject(),
                            graph_datum.GetEdgeWeightDeviceObject(),
			    graph_datum.m_wl_bitmap_out_high.DeviceObject(),
                            graph_datum.m_wl_bitmap_in.DeviceObject()
			    );
                    break;
                default:
                    assert(false);
            }
            stream.Sync();
        }
    }
}
#endif //HYBRID_ALGO_VARIANTS_H
