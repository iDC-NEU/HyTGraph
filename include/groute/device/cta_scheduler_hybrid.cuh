// Groute: An Asynchronous Multi-GPU Programming Framework
// http://www.github.com/groute/groute
// Copyright (c) 2017, A. Barak
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the names of the copyright holders nor the names of its 
//   contributors may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef __GROUTE_CTA_SCHEDULER_HYBRID_H
#define __GROUTE_CTA_SCHEDULER_HYBRID_H

#include <initializer_list>
#include <vector>
#include <map>
#include <memory>
#include <cuda_runtime.h>
#include <mutex>
#include <cub/util_ptx.cuh>
#include <groute/device/cta_scheduler.cuh>

//#if __CUDA_ARCH__ >= 700
//#define NO_CTA_WARP_INTRINSICS
//#endif

#define HYBRID_THRESHOLD 0

namespace groute
{
    namespace dev
    {
        enum LBMode
        {
            LB_COARSE_GRAINED,
            LB_FINE_GRAINED,
            LB_HYBRID,
        };

        template<typename TMetaData, LBMode Mode = LB_COARSE_GRAINED>
        struct CTAWorkSchedulerNew
        {
            template<typename TWork>
            __device__ __forceinline__ static void schedule(np_local<TMetaData> &np_local, TWork work, bool zcflag)
            {
                const int WP_SIZE = CUB_PTX_WARP_THREADS;
                const int TB_SIZE = blockDim.x;

                const int NP_WP_CROSSOVER = 1;
                const int NP_TB_CROSSOVER = blockDim.x;

#ifdef NO_CTA_WARP_INTRINSICS
                typedef union np_shared<empty_np, tb_np<TMetaData>, empty_np> np_shared_type;
#else
                typedef union np_shared<empty_np, tb_np<TMetaData>, warp_np<32, TMetaData>> np_shared_type; // 32 is max number of warps in block
#endif

                __shared__ np_shared_type np_shared;

                if (threadIdx.x == 0)
                {
                    np_shared.tb.owner = TB_SIZE + 1;
                }

                __syncthreads();

                //
                // First scheduler: processing high-degree work items using the entire block
                //
                while (true)
                {
                    if (np_local.size >= NP_TB_CROSSOVER)
                    {
                        // 'Elect' one owner for the entire thread block
                        np_shared.tb.owner = threadIdx.x;
                    }

                    __syncthreads();

                    if (np_shared.tb.owner == TB_SIZE + 1)
                    {
                        // No owner was elected, i.e. no high-degree work items remain

#ifdef NO_CTA_WARP_INTRINSICS
                        // No need to sync threads before moving on to WP scheduler
                        // because it does not use shared memory
#else
                        __syncthreads(); // Necessary do to the shared memory union used by both TB and WP schedulers
#endif
                        break;
                    }

                    if (np_shared.tb.owner == threadIdx.x)
                    {
                        // This thread is the owner
                        np_shared.tb.start = np_local.start;
                        np_shared.tb.size = np_local.size;
                        np_shared.tb.meta_data = np_local.meta_data;

                        // Mark this work-item as processed for future schedulers
                        np_local.start = 0;
                        np_local.size = 0;
                    }

                    __syncthreads();

                    uint64_t start_org = np_shared.tb.start;
		            uint64_t start = start_org & (~(0x1fULL));
                    uint64_t size = np_shared.tb.size;
                    TMetaData meta_data = np_shared.tb.meta_data;

                    if (np_shared.tb.owner == threadIdx.x)
                    {
                        np_shared.tb.owner = TB_SIZE + 1;
                    }

                    // Use all threads in thread block to execute individual work
                    if(!zcflag){
			             for (uint64_t ii = threadIdx.x; ii < size; ii += TB_SIZE)
			             {
			                 work(start_org + ii, meta_data);
			             }
		            }
		            else{
			             for (uint64_t ii = threadIdx.x + start; ii < start_org + size; ii += TB_SIZE)
			             {
			                     if(ii >= start_org){
				                        work(ii, meta_data);
			                     }
			             }
		            }
                    __syncthreads();
                }

                //
                // Second scheduler: tackle medium-degree work items using the warp
                //

#ifndef NO_CTA_WARP_INTRINSICS
                const int warp_id = threadIdx.x / WP_SIZE; 
#endif
                const int lane_id = cub::LaneId();

                while (__any_sync(0xffffffff, np_local.size >= NP_WP_CROSSOVER)) // any thread satisfied size > 1 ,warp need execute
                {
#ifdef NO_CTA_WARP_INTRINSICS
                    // Compete for work scheduling
                    int mask = __ballot_sync(0xffffffff, np_local.size >= NP_WP_CROSSOVER ? 1 : 0);
                    // Select a deterministic winner
                    int leader = __ffs(mask) - 1;

                    // Broadcast data from the leader
                    index_type start = cub::ShuffleIndex<WP_SIZE>(np_local.start, leader, 0xffffffff);
                    index_type size = cub::ShuffleIndex<WP_SIZE>(np_local.size, leader, 0xffffffff);
                    TMetaData meta_data = cub::ShuffleIndex<WP_SIZE>(np_local.meta_data, leader, 0xffffffff);

                    if (leader == lane_id)
                    {
                        // Mark this work-item as processed
                        np_local.start = 0;
                        np_local.size = 0;
                    }
#else
                    if (np_local.size >= NP_WP_CROSSOVER)
                    {
                        // Again, race to select an owner for warp
                        np_shared.warp.owner[warp_id] = lane_id;
                    }

                    cub::WARP_SYNC(0xffffffff);

                    if (np_shared.warp.owner[warp_id] == lane_id)
                    {
                        // This thread is owner
                        np_shared.warp.start[warp_id] = np_local.start;
                        np_shared.warp.size[warp_id] = np_local.size;
                        np_shared.warp.meta_data[warp_id] = np_local.meta_data;

                        // Mark this work-item as processed
                        np_local.start = 0;
                        np_local.size = 0;
                    }

                    cub::WARP_SYNC(0xffffffff);
		    
		            //Align starting index to 128-byte boundary
		            uint64_t start_org = np_shared.warp.start[warp_id];
                    uint64_t start = start_org & (~(0x1fULL));  //4-byte data type
                    //printf("start_org:%d,start:%d\n",start_org,start);
                    uint64_t size = np_shared.warp.size[warp_id];
 		            //printf("size:%d\n",size);
                    TMetaData meta_data = np_shared.warp.meta_data[warp_id];
#endif

		            if (!zcflag){
			             for (uint64_t ii = lane_id; ii < size; ii += WP_SIZE)
			             {	 
                              work(start_org + ii, meta_data);
			             }
		            }
		            else {
		                 for (uint64_t ii = start + lane_id; ii < start_org + size; ii += WP_SIZE)
                         { 
			                 if(ii >= start_org){
			                     work(ii , meta_data);
			                 }
                          }
		             }
                    //cub::WARP_SYNC(0xffffffff);
                }

                __syncthreads();
                
                
                assert(np_local.size>0);
                //
                // Third scheduler: tackle all work-items with size < 32 serially
                //
                // We did not implement the FG (Finegrained) scheduling for simplicity
                // It is possible to disable this scheduler by setting NP_WP_CROSSOVER to 0
                 //for (int ii = 0; ii < np_local.size; ii++)
                 //{   assert(false); 
                 //    work(np_local.start + ii, np_local.meta_data);
                 //}
            }
        };
    }
}

#endif // __GROUTE_CTA_SCHEDULER_HYBRID_H
