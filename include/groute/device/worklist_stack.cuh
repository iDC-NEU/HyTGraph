// ----------------------------------------------------------------
// SEP-Graph: Finding Shortest Execution Paths for Graph Processing under a Hybrid Framework on GPU
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE
// in the root directory of this source distribution.
// ----------------------------------------------------------------
#ifndef HYBRID_WORKLIST_STACK_H
#define HYBRID_WORKLIST_STACK_H

#include <groute/device/queue.cuh>
#include <utils/graphs/traversal.h>
#include <cub/grid/grid_barrier.cuh>

namespace sepgraph {
    namespace dev {
        template<typename T>
        struct WorklistStack {
            uint32_t *m_stack;
            T *m_data;
            uint32_t *m_stack_depth;
            uint32_t *m_data_pos;
            uint32_t m_max_depth;
            uint32_t m_capacity;

            __host__ __device__

            WorklistStack(uint32_t *stack, T *data, uint32_t *stack_depth, uint32_t *data_pos, uint32_t max_depth,
                          uint32_t capacity) :
                    m_stack(stack), m_data(data), m_stack_depth(stack_depth), m_data_pos(data_pos),
                    m_max_depth(max_depth), m_capacity(capacity) {
#if defined(__CUDA_ARCH__)
                assert(*m_stack_depth == 0);
                assert(*m_data_pos == 0);
#endif
                assert(m_max_depth > 0);
                assert(m_capacity > 0);
            }

            __device__ __forceinline__

            void append(T item) {
                uint32_t last_pos = atomicAdd(m_data_pos, 1);

                assert(last_pos < m_capacity);
                m_data[last_pos] = item;
            }

            __device__ __forceinline__

            void push() {
                assert(*m_stack_depth < m_max_depth);

                *m_stack_depth += 1;
                m_stack[*m_stack_depth] = *m_data_pos;
            }

            __device__ __forceinline__

            void reset() {
                *m_stack_depth = 0;
                *m_data_pos = 0;
            }

            __device__ __forceinline__

            uint32_t begin_pos(uint32_t depth) const {
                assert(depth >= 0);
                assert(depth < *m_stack_depth);

                return m_stack[depth];
            }

            __device__ __forceinline__

            uint32_t end_pos(uint32_t depth) const {
                assert(depth >= 0);
                assert(depth + 1 <= *m_stack_depth);

                return m_stack[depth + 1];
            }

            __device__ __forceinline__

            T read(uint32_t pos) const {
                assert(pos >= 0);
                assert(pos < *m_data_pos);

                return m_data[pos];
            }

            __device__ __forceinline__

            uint32_t get_depth() const {
                return *m_stack_depth;
            }

            __device__ __forceinline__

            uint32_t count() const {
                return *m_data_pos;
            }

            __device__ __forceinline__

            uint32_t count(uint32_t depth) const {
                assert(depth >= 0);
                assert(depth + 1 <= *m_stack_depth);

                return m_stack[depth + 1] - m_stack[depth];
            }
        };

        template<typename T>
        __global__ void Pushback(WorklistStack<T> worklist_stack,
                                 T *p_items,
                                 uint32_t len) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t s_top = *worklist_stack.m_stack_depth;
            uint32_t last_pos = worklist_stack.m_stack[s_top];

            if (tid == 0) {
                assert(last_pos + len <= worklist_stack.m_capacity);
                assert(last_pos == *worklist_stack.m_data_pos);
            }

            for (uint32_t idx = tid; idx < len; idx += nthreads) {
                worklist_stack.m_data[last_pos + idx] = p_items[idx];
            }
        }

        // We need a barrier for Pushback function
        template<typename T>
        __global__ void Commit(WorklistStack<T> worklist_stack, uint32_t len) {
            uint32_t tid = TID_1D;

            if (tid == 0) {
                uint32_t s_top = *worklist_stack.m_stack_depth;
                uint32_t last_pos = worklist_stack.m_stack[s_top];

                *worklist_stack.m_data_pos += len;
                *worklist_stack.m_stack_depth += 1;
                worklist_stack.m_stack[*worklist_stack.m_stack_depth] = last_pos + len;
            }
        }
    }


    template<typename T>
    class WorklistStack {
        uint32_t *m_stack;
        T *m_data;
        uint32_t *m_stack_depth;
        uint32_t *m_data_pos;
        uint32_t m_max_depth;
        uint32_t m_capacity;
    public:
        WorklistStack(uint32_t capacity, uint32_t max_depth = 100000) :
                m_capacity(capacity), m_max_depth(max_depth) {
            assert(m_max_depth > 0);
            assert(m_capacity > 0);

            GROUTE_CUDA_CHECK(cudaMalloc((void **) &m_stack, sizeof(uint32_t) * (m_max_depth + 1)));
            GROUTE_CUDA_CHECK(cudaMalloc((void **) &m_data, sizeof(T) * m_capacity));
            GROUTE_CUDA_CHECK(cudaMalloc((void **) &m_stack_depth, sizeof(uint32_t)));
            GROUTE_CUDA_CHECK(cudaMalloc((void **) &m_data_pos, sizeof(uint32_t)));
            GROUTE_CUDA_CHECK(cudaMemset(m_stack_depth, 0, sizeof(uint32_t)));
            GROUTE_CUDA_CHECK(cudaMemset(m_data_pos, 0, sizeof(uint32_t)));
        }

        typedef dev::WorklistStack<T> DeviceObjectType;

        DeviceObjectType DeviceObject() const {
            return dev::WorklistStack<T>(m_stack,
                                         m_data,
                                         m_stack_depth,
                                         m_data_pos,
                                         m_max_depth,
                                         m_capacity);
        }

        void ResetAsync(cudaStream_t stream) {
            GROUTE_CUDA_CHECK(cudaMemset(m_stack_depth, 0, sizeof(uint32_t)));
            GROUTE_CUDA_CHECK(cudaMemset(m_data_pos, 0, sizeof(uint32_t)));
        }

        void ResetAsync(const groute::Stream &stream) {
            ResetAsync(stream.cuda_stream);
        }

        void PushAsync(T *data_ptr, uint32_t len, const groute::Stream &stream) {
            dim3 grid_dims, block_dims;

            KernelSizing(grid_dims, block_dims, len);

            dev::Pushback << < grid_dims, block_dims, 0, stream.cuda_stream >> > (DeviceObject(), data_ptr, len);
            dev::Commit << < 1, 1, 0, stream.cuda_stream >> > (DeviceObject(), len);
        }

        T *GetDataPtr(uint32_t depth) {
            uint32_t begin_pos;

            GROUTE_CUDA_CHECK(cudaMemcpy(&begin_pos, m_stack + depth, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            return m_data + begin_pos;
        }

        uint32_t GetDepth(const groute::Stream &stream) const {
            uint32_t depth;

            GROUTE_CUDA_CHECK(cudaMemcpy(&depth, m_stack_depth, sizeof(uint32_t), cudaMemcpyDeviceToHost));

            return depth;
        }

        uint32_t GetCount(const groute::Stream &stream) const {
            uint32_t count;

            GROUTE_CUDA_CHECK(cudaMemcpy(&count, m_data_pos, sizeof(uint32_t), cudaMemcpyDeviceToHost));

            return count;
        }

        uint32_t GetCount(uint32_t depth, const groute::Stream &stream) const {
            uint32_t begin_pos, end_pos;

            GROUTE_CUDA_CHECK(cudaMemcpy(&begin_pos, m_stack + depth, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            GROUTE_CUDA_CHECK(cudaMemcpy(&end_pos, m_stack + depth + 1, sizeof(uint32_t), cudaMemcpyDeviceToHost));

            return end_pos - begin_pos;
        }

        ~WorklistStack() {
            GROUTE_CUDA_CHECK(cudaFree(m_stack));
            GROUTE_CUDA_CHECK(cudaFree(m_data));
            GROUTE_CUDA_CHECK(cudaFree(m_stack_depth));
            GROUTE_CUDA_CHECK(cudaFree(m_data_pos));
            m_stack = nullptr;
            m_data = nullptr;
            m_stack_depth = nullptr;
            m_data_pos = nullptr;
            m_max_depth = 0;
            m_capacity = 0;
        }
    };
}
#endif //HYBRID_WORKLIST_STACK_H
