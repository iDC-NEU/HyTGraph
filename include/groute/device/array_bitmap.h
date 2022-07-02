// ----------------------------------------------------------------
// SEP-Graph: Finding Shortest Execution Paths for Graph Processing under a Hybrid Framework on GPU
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE
// in the root directory of this source distribution.
// ----------------------------------------------------------------
#ifndef GROUTE_ARRAY_BITMAP_H
#define GROUTE_ARRAY_BITMAP_H
#define ELE_TYPE int

namespace sepgraph
{
    namespace dev
    {
        class ArrayBitmap
        {
        private:
            ELE_TYPE *const m_data;
            uint32_t *const m_positive_count;
            const uint32_t m_size;
            //uint32_t *const m_positive_count_partition;
            //const uint32_t partitions;

        public:
            __host__ __device__

            ArrayBitmap(ELE_TYPE *data, uint32_t size, uint32_t *positive_count) : m_data(data),
                                                                                   m_size(size),
                                                                                   m_positive_count(positive_count)
            {
                assert(sizeof(unsigned long long int) == 8);
            }

            __device__ __forceinline__

                void
                set_bit_atomic(uint32_t pos)
            {
                assert(pos < m_size);

                if (atomicExch(m_data + pos, 1) == 0)
                {
                    atomicAdd(m_positive_count, 1);
                }
            }

            //            __device__ __forceinline__
            //
            //            void set_bit_atomic(uint32_t pos) {
            //                assert(pos < m_size);
            //
            //                m_data[pos] = 1;
            //            }

            __device__ __forceinline__

                void
                set_bit(uint32_t pos)
            {
                assert(pos < m_size);

                if (m_data[pos] == 0)
                {
                    m_data[pos] = 1;
                    *m_positive_count = *m_positive_count + 1;
                }
            }

            __device__ __forceinline__

                void
                reset()
            {
                uint32_t tid = TID_1D;
                uint32_t nthreads = TOTAL_THREADS_1D;

                for (int i = 0 + tid; i < m_size; i += nthreads)
                    m_data[i] = 0;

                if (tid == 0)
                {
                    *m_positive_count = 0;
                }
            }

            __device__ __forceinline__

                bool
                get_bit(uint32_t pos)
            {
                return m_data[pos];
            }

            __device__ __forceinline__

                uint32_t
                get_size() const
            {
                return m_size;
            }

            __device__ __forceinline__

                uint32_t
                get_positive_count()
            {
                return *m_positive_count;
            }
            // __device__ __forceinline__

            //     uint32_t
            //     get_positive_count_partition(index_t partition_id)
            // {
            //     return m_positive_count_partition[partition_id];
            // }

            __device__ __forceinline__

                bool
                operator==(const sepgraph::dev::ArrayBitmap &other) const
            {
                return m_data == other.m_data &&
                       m_positive_count == other.m_positive_count &&
                       m_size == other.m_size;
            }

            __device__ __forceinline__

                bool
                operator!=(const sepgraph::dev::ArrayBitmap &other) const
            {
                return !this->operator==(other);
            }
        };

        __global__ void GetPositiveCount(ELE_TYPE *const p_data, uint32_t size, uint32_t *p_positive_count)
        {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t local_sum = 0;

            for (uint32_t i = 0 + tid; i < size; i += nthreads)
            {
                if (p_data[i])
                    local_sum++;
            }

            if (local_sum > 0)
                atomicAdd(p_positive_count, local_sum);
        }
        // __global__ void GetPositiveCountRange(ELE_TYPE *const p_data, uint32_t *p_positive_count, index_t start, index_t end, index_t partition_id)
        // {
        //     uint32_t tid = TID_1D;
        //     uint32_t nthreads = TOTAL_THREADS_1D;
        //     uint32_t local_sum = 0;

        //     for (uint32_t i = 0 + tid; i < end - start; i += nthreads)
        //     {
        //         uint32_t position = i + start;
        //         if (p_data[position])
        //             local_sum++;
        //     }

        //     if (local_sum > 0)
        //         atomicAdd(p_positive_count[partition_id], local_sum);
        // }
    } // namespace dev

    class ArrayBitmap
    {
    private:
        ELE_TYPE *m_data;
        uint32_t m_size;
        uint32_t *m_positive_count;
        uint32_t *m_host_positive_count;

        //uint32_t *m_positive_count_partition;
        //uint32_t *m_host_positive_count_partition;
        //uint32_t partitions;

        ArrayBitmap &operator=(const ArrayBitmap &other) = default;

        void Alloc()
        {
            if (m_size == 0)
                return;
            GROUTE_CUDA_CHECK(cudaMalloc(&m_data, sizeof(ELE_TYPE) * m_size));
            GROUTE_CUDA_CHECK(cudaMalloc(&m_positive_count, sizeof(uint32_t)));
            GROUTE_CUDA_CHECK(cudaMemset(m_positive_count, 0, sizeof(uint32_t)));
            GROUTE_CUDA_CHECK(cudaMallocHost(&m_host_positive_count, sizeof(uint32_t)));
        }
        // void AllocP()
        // {
        //     if (m_size == 0)
        //         return;
        //     GROUTE_CUDA_CHECK(cudaMalloc(&m_data, sizeof(ELE_TYPE) * m_size));
        //     GROUTE_CUDA_CHECK(cudaMalloc(&m_positive_count, sizeof(uint32_t)));
        //     GROUTE_CUDA_CHECK(cudaMemset(m_positive_count, 0, sizeof(uint32_t)));
        //     GROUTE_CUDA_CHECK(cudaMalloc(&m_positive_count_partition, partitions * sizeof(uint32_t)));
        //     GROUTE_CUDA_CHECK(cudaMemset(m_positive_count_partition, 0, partitions * sizeof(uint32_t)));

        //     GROUTE_CUDA_CHECK(cudaMallocHost(&m_host_positive_count, sizeof(uint32_t)));
        //     GROUTE_CUDA_CHECK(cudaMallocHost(&m_host_positive_count_partition, sizeof(uint32_t) * partitions));
        // }

        void Free()
        {
            if (m_size == 0)
                return;
            GROUTE_CUDA_CHECK(cudaFree(m_data));
            GROUTE_CUDA_CHECK(cudaFree(m_positive_count));
            GROUTE_CUDA_CHECK(cudaFreeHost(m_host_positive_count));
        }

    public:
        typedef dev::ArrayBitmap DeviceObjectType;

        ArrayBitmap() : m_data(nullptr), m_size(0)
        {
        }

        ArrayBitmap(uint32_t size) : m_size(size)
        {
            Alloc();
        }

        ArrayBitmap(const ArrayBitmap &other) = delete;

        ArrayBitmap(ArrayBitmap &&other) = delete;

        ArrayBitmap &operator=(ArrayBitmap &&other)
        {
            *this = other;
            new (&other) ArrayBitmap();

            return *this;
        }

        ~ArrayBitmap()
        {
            Free();
        }

        DeviceObjectType DeviceObject() const
        {
            assert(m_size > 0);
            return dev::ArrayBitmap(m_data, m_size, m_positive_count);
        }

        void ResetAsync(cudaStream_t stream)
        {
            assert(m_size > 0);

            GROUTE_CUDA_CHECK(cudaMemsetAsync(m_data, 0, sizeof(ELE_TYPE) * m_size, stream));
            GROUTE_CUDA_CHECK(cudaMemsetAsync(m_positive_count, 0, sizeof(uint32_t), stream));
        }

        void ResetAsync(const groute::Stream &stream)
        {
            ResetAsync(stream.cuda_stream);
        }

        // void ResetAsyncRange(cudaStream_t stream, index_t start, index_t end, index_t partition_id)
        // {
        //     assert(m_size > 0);
        //     GROUTE_CUDA_CHECK(cudaMemsetAsync(m_data + start, 0, sizeof(ELE_TYPE) * (end - start), stream));
        //     GROUTE_CUDA_CHECK(cudaMemsetAsync(m_positive_count, 0, sizeof(uint32_t), stream));
        // }

        // void ResetAsyncRange(const groute::Stream &stream, index_t start, index_t end, index_t partition_id)
        // {
        //     ResetAsyncRange(stream.cuda_stream, start, end, partition_id);
        // }

        void Swap(ArrayBitmap &other)
        {
            std::swap(m_data, other.m_data);
            std::swap(m_size, other.m_size);
            std::swap(m_positive_count, other.m_positive_count);
            std::swap(m_host_positive_count, other.m_host_positive_count);
        }

        uint32_t GetSize() const
        {
            return m_size;
        }

        uint32_t GetPositiveCount(const groute::Stream &stream)
        {
            GROUTE_CUDA_CHECK(cudaMemcpyAsync(m_host_positive_count,
                                              m_positive_count,
                                              sizeof(uint32_t),
                                              cudaMemcpyDeviceToHost,
                                              stream.cuda_stream));
            stream.Sync();

            assert(*m_host_positive_count <= m_size);

            return *m_host_positive_count;
            //            dim3 grid_dims, block_dims;
            //
            //            KernelSizing(grid_dims, block_dims, m_size);
            //
            //            Stopwatch sw(true);
            //
            //            cudaMemsetAsync(m_positive_count, 0, sizeof(uint32_t), stream.cuda_stream);
            //            dev::GetPositiveCount<< < grid_dims, block_dims, 0, stream.cuda_stream >> > (m_data,
            //                    m_size, m_positive_count);
            //            cudaMemcpyAsync(m_host_positive_count, m_positive_count, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream.cuda_stream);
            //            stream.Sync();
            //
            //            sw.stop();
            //            printf("GetPos: %f\n", sw.ms());

            return *m_host_positive_count;
        }
    };
} // namespace sepgraph

#endif //GROUTE_ARRAY_BITMAP_H
