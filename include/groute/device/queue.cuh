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

#ifndef __GROUTE_WORK_QUEUE_H
#define __GROUTE_WORK_QUEUE_H

#include <initializer_list>
#include <vector>
#include <map>
#include <memory>
#include <cuda_runtime.h>
#include <mutex>
#include <new> // Used for the in-memory ctor call in the move assignment operator below  
#include <device_launch_parameters.h>
#include <cub/util_ptx.cuh>
#include <groute/common.h>
#include <groute/graphs/common.h>
#include <groute/device/work_source.cuh>
//
// Common device-related MACROS
//

// Default block size for system kernels  
#define GROUTE_BLOCK_THREADS 256
#define TID_1D (threadIdx.x + blockIdx.x * blockDim.x)
#define TOTAL_THREADS_1D (gridDim.x * blockDim.x)
//
//
//

namespace groute {
    namespace dev {

        __device__ __forceinline__ void warp_active_count(int &first, int &offset, int &total) {
            unsigned int active = __ballot_sync(0xffffffff, 1);
            total = __popc(active);
            offset = __popc(active & cub::LaneMaskLt());
            first = __ffs(active) - 1;
        }

        //
        // Queue classes (device):
        //

        /*
        * @brief A device-level Queue (see host controller object below for usage)
        */
        template<typename T>
        class Queue {
            T *m_data;
            uint32_t *m_count;
            uint32_t m_capacity;

        public:
            __host__ __device__ Queue(T *data, uint32_t *count, uint32_t capacity) :
                    m_data(data), m_count(count), m_capacity(capacity) {}

            __device__ __forceinline__ void append(const T &item) const {
                uint32_t allocation = atomicAdd(m_count, 1); // Just a naive atomic add
                assert(allocation < m_capacity);
                m_data[allocation] = item;
            }

            __device__ void append_warp(const T &item) const {
                int leader, total, offset;
                uint32_t allocation = 0;

                warp_active_count(leader, offset, total);

                if (offset == 0) {
                    allocation = atomicAdd((uint32_t *) m_count, total);
                    assert(allocation + total <= m_capacity);
                }

                allocation = cub::ShuffleIndex<32>(allocation, leader, 0xffffffff);
                m_data[allocation + offset] = item;
            }

            __device__ void append_warp(const T &item, int leader, int total, int offset) const {
                uint32_t allocation = 0;

                if (offset == 0) // The leader thread  
                {
                    allocation = atomicAdd((uint32_t *) m_count, total);
                    assert(allocation + total <= m_capacity);
                }

                allocation = cub::ShuffleIndex<32>(allocation, leader, 0xffffffff);
                m_data[allocation + offset] = item;
            }

            __device__ __forceinline__ void reset() const {
                *m_count = 0;
            }

            __device__ __forceinline__ T read(int i) const {
                return m_data[i];
            }

            __device__ __forceinline__ uint32_t count() const {
                return *m_count;
            }

            __device__ __forceinline__ void pop(uint32_t count) const {
                assert(*m_count >= count);
                *m_count -= count;
            }

            __device__ __forceinline__  T *data_ptr() { return m_data; }

            __device__ __forceinline__ bool operator==(const groute::dev::Queue<T> &other) const {
                return m_data == other.m_data &&
                       m_capacity == other.m_capacity &&
                       m_count == other.m_count;
            }

            __device__ __forceinline__ bool operator!=(const groute::dev::Queue<T> &other) const {
                return !this->operator==(other);
            }
        };
    }


    // 
    // Queue control kernels:  
    //

    namespace queue {
        namespace kernels {

            template<typename T>
            __global__ void QueueReset(dev::Queue<T> queue) {
                if (threadIdx.x == 0 && blockIdx.x == 0)
                    queue.reset();
            }

            static __global__ void ResetCounters(uint32_t *counters, uint32_t num_counters) {
                if (TID_1D < num_counters)
                    counters[TID_1D] = 0;
            }

            template<typename T>
            __global__ void QueueAppendItem(dev::Queue<T> queue, T item) {
                if (threadIdx.x == 0 && blockIdx.x == 0)
                    queue.append(item);
            }

            template<typename T, template<typename> class WorkSource>
            __global__ void QueueAppendItems(dev::Queue<T> queue, WorkSource<T> work_source) {
                auto tid = TID_1D;
                auto nthreads = TOTAL_THREADS_1D;

                for (int i = 0 + tid; i < work_source.get_size(); i += nthreads) {
                    T item = work_source.get_work(i);
                    queue.append(item);
                }
            }

        }
    }

    //
    // Queue control classes (host):  
    //

    /*
    * @brief Host controller object for dev::Queue (see above)
    */
    template<typename T>
    class Queue {
        enum {
            NUM_COUNTERS = 32
        }; // Number of counter slots

        // device buffer / counters 
        T *m_data;
        uint32_t *m_counters;

        // Pinned host counter
        uint32_t *m_host_count;

        uint32_t m_capacity;
        int32_t m_current_slot; // The currently used counter slot (see ResetAsync method)

        bool m_mem_owner;

        Queue &operator=(const Queue &other) = default; // For private use only

        void Alloc(Endpoint endpoint, const char *name) {
            if (m_capacity == 0) return;

            if (m_mem_owner)
            GROUTE_CUDA_CHECK(cudaMalloc(&m_data, sizeof(T) * m_capacity));
            GROUTE_CUDA_CHECK(cudaMalloc(&m_counters, NUM_COUNTERS * sizeof(uint32_t)));
            GROUTE_CUDA_CHECK(cudaMallocHost(&m_host_count, sizeof(uint32_t)));
        }

        void Free() {
            if (m_capacity == 0) return;

            if (m_mem_owner)
                GROUTE_CUDA_CHECK(cudaFree(m_data));
            GROUTE_CUDA_CHECK(cudaFree(m_counters));
            GROUTE_CUDA_CHECK(cudaFreeHost(m_host_count));
        }

    public:
        Queue(uint32_t capacity = 0, Endpoint endpoint = Endpoint(), const char *name = "") :
                m_data(nullptr), m_mem_owner(true), m_counters(nullptr), m_capacity(capacity), m_current_slot(-1) {
            Alloc(endpoint, name);
        }

        Queue(T *mem_buffer, uint32_t mem_size, Endpoint endpoint = Endpoint(), const char *name = "") :
                m_data(mem_buffer), m_mem_owner(false), m_counters(nullptr), m_capacity(mem_size), m_current_slot(-1) {
            Alloc(endpoint, name);
        }

        Queue(const Queue &other) = delete;

        Queue(Queue &&other) {
            *this = std::move(other);
        }

        Queue &operator=(Queue &&other) {
            *this = other;           // First copy all fields  
            new(&other) Queue(0);   // Clear up other

            return (*this);
        }

        void Swap(Queue &other) {
            std::swap(m_data, other.m_data);
            std::swap(m_counters, other.m_counters);
            std::swap(m_host_count, other.m_host_count);
            std::swap(m_capacity, other.m_capacity);
            std::swap(m_current_slot, other.m_current_slot);
            std::swap(m_mem_owner, other.m_mem_owner);
        }

        ~Queue() {
            Free();
        }

        T *GetDeviceDataPtr() const { return m_data; }

        typedef dev::Queue<T> DeviceObjectType;

        DeviceObjectType DeviceObject() const {
            assert(m_current_slot >= 0 && m_current_slot < NUM_COUNTERS);
            return dev::Queue<T>(m_data, m_counters + m_current_slot, m_capacity);
        }

        void ResetAsync(cudaStream_t stream) {
            //
            // We use multiple counter slots to avoid running a kernel each time a reset is required  
            //

            m_current_slot = (m_current_slot + 1) % NUM_COUNTERS;
            if (m_current_slot == 0) {
                queue::kernels::ResetCounters << < 1, NUM_COUNTERS, 0, stream >> > (m_counters, NUM_COUNTERS);
            }
        }

        void AppendItemAsync(cudaStream_t stream, const T &item) const {
            queue::kernels::QueueAppendItem << < 1, 1, 0, stream >> > (DeviceObject(), item);
        }

        template<template<typename> class WorkSource>
        void AppendItemsAsync(cudaStream_t stream, const WorkSource<T> &work_source) {
            dim3 grid_dims, block_dims;

            KernelSizing(grid_dims, block_dims, work_source.get_size());
            queue::kernels::QueueAppendItems << < grid_dims, block_dims, 0, stream >> > (DeviceObject(), work_source);
        }

        void ResetAsync(const Stream &stream) {
            ResetAsync(stream.cuda_stream);
        }

        void AppendItemAsync(const Stream &stream, const T &item) const {
            AppendItemAsync(stream.cuda_stream, item);
        }

        template<template<typename> class WorkSource>
        void AppendItemsAsync(const Stream &stream, const WorkSource<T> &work_source) {
            AppendItemsAsync(stream.cuda_stream, work_source);
        }

        uint32_t GetCount(const Stream &stream) const {
            assert(m_current_slot >= 0 && m_current_slot < NUM_COUNTERS);
            GROUTE_CUDA_CHECK(
                    cudaMemcpyAsync(m_host_count, m_counters + m_current_slot, sizeof(uint32_t), cudaMemcpyDeviceToHost,
                                    stream.cuda_stream));
            stream.Sync();
            return *m_host_count;
        }

        void SetLength(const Stream &stream, uint32_t new_length) {
            assert(m_current_slot >= 0 && m_current_slot < NUM_COUNTERS);
            assert(new_length <= m_capacity);

            *m_host_count = new_length;

            GROUTE_CUDA_CHECK(
                    cudaMemcpyAsync(m_counters + m_current_slot, m_host_count, sizeof(uint32_t), cudaMemcpyHostToDevice,
                                    stream.cuda_stream));
            stream.Sync();
        }

        void PrintOffsets(const Stream &stream) const {
            printf("\nQueue (Debug): count: %u (capacity: %u)",
                   GetCount(stream), m_capacity);
        }

        groute::dev::WorkSourceArray<index_t> ToWorkSource(const Stream &stream) {
            return groute::dev::WorkSourceArray<index_t>(GetDeviceDataPtr(), GetCount(stream));
        }
    };
}

#endif // __GROUTE_WORK_QUEUE_H
