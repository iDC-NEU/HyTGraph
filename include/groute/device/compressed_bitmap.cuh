// ----------------------------------------------------------------
// SEP-Graph: Finding Shortest Execution Paths for Graph Processing under a Hybrid Framework on GPU
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE
// in the root directory of this source distribution.
// ----------------------------------------------------------------
#ifndef HYBRID_COMPRESSED_BITMAP_H
#define HYBRID_COMPRESSED_BITMAP_H

#include <groute/device/queue.cuh>
#include <groute/device/work_source.cuh>
#include <utils/graphs/traversal.h>

namespace sepgraph {
    namespace dev {
        class CompressedBitmap {
        public:
            __host__ __device__
            CompressedBitmap(uint64_t *data, size_t size, size_t *positive_count) : m_data(data),
                                                                                    m_size(size),
                                                                                    m_positive_count(positive_count),
                                                                                    m_num_words(
                                                                                            (size + kBitsPerWord - 1) /
                                                                                            kBitsPerWord) {
                assert(sizeof(unsigned long long int) == 8);
                assert(sizeof(size_t) == sizeof(unsigned long long int));
            }

            __device__ __forceinline__
            void set_bit(size_t pos) {
                m_data[word_offset(pos)] |= ((uint64_t) 1l << bit_offset(pos));
            }

            __device__ __forceinline__
            void set_bit_atomic(size_t pos) {
                uint64_t old_val, new_val;
                do {
                    old_val = m_data[word_offset(pos)];
                    if (old_val & ((uint64_t) 1l << bit_offset(pos)))
                        return;
                    new_val = old_val | ((uint64_t) 1l << bit_offset(pos));
                } while (old_val !=
                         atomicCAS(reinterpret_cast<unsigned long long int *>(m_data + word_offset(pos)), old_val,
                                   new_val));
                if ((old_val & (1l << bit_offset(pos))) == 0) {
                    atomicAdd(reinterpret_cast<unsigned long long int *>(m_positive_count), 1);
                }
            }

            __device__ __forceinline__
            void reset() {
                uint32_t tid = TID_1D;
                uint32_t nthreads = TOTAL_THREADS_1D;

                for (int i = 0 + tid; i < m_num_words; i += nthreads)
                    m_data[i] = 0;

                if (tid == 0) {
                    *m_positive_count = 0;
                }
            }

            __device__ __forceinline__
            bool get_bit(size_t pos) {
                return (m_data[word_offset(pos)] >> bit_offset(pos)) & 1l;
            }

            __device__ __forceinline__
            size_t get_size() const {
                return m_size;
            }

            __device__ __forceinline__
            size_t get_positive_count() {
                return *m_positive_count;
            }

            __device__ __forceinline__ bool operator==(const sepgraph::dev::CompressedBitmap &other) const {
                return m_data == other.m_data &&
                       m_positive_count == other.m_positive_count &&
                       m_size == other.m_size;
            }

            __device__ __forceinline__ bool operator!=(const sepgraph::dev::CompressedBitmap &other) const {
                return !this->operator==(other);
            }

        private:
            uint64_t *const m_data;
            size_t *const m_positive_count;
            const size_t m_size;
            const uint64_t m_num_words;

            static const uint32_t kBitsPerWord = 64;

            __device__ __forceinline__

            uint64_t word_offset(size_t n) { return n / kBitsPerWord; }

            __device__ __forceinline__

            uint64_t bit_offset(size_t n) { return n & (kBitsPerWord - 1); }

        };

        namespace kernels {
            __global__ void SetBit(dev::CompressedBitmap bitmap, index_t pos) {
                if (threadIdx.x == 0 && blockIdx.x == 0)
                    bitmap.set_bit_atomic(pos);
            }

            __global__ void SetBits(dev::CompressedBitmap bitmap, groute::dev::WorkSourceArray<index_t> work_source) {
                auto tid = TID_1D;
                auto nthreads = TOTAL_THREADS_1D;

                for (int i = 0 + tid; i < work_source.get_size(); i += nthreads) {
                    auto work = work_source.get_work(i);
                    bitmap.set_bit_atomic(work);
                }
            }
        }
    }


    class CompressedBitmap {
    public:
        typedef dev::CompressedBitmap DeviceObjectType;

        CompressedBitmap() : m_data(nullptr),
                             m_size(0),
                             m_host_positive_count(nullptr),
                             m_positive_count(nullptr) {

        }

        CompressedBitmap(size_t size) : m_size(size) {
            Alloc();
        }

        CompressedBitmap(const CompressedBitmap &other) = delete;

        CompressedBitmap(CompressedBitmap &&other) {
//            this->m_positive_count = other.m_positive_count;
//            this->m_size = other.m_size;
//            this->m_host_positive_count = other.m_host_positive_count;
//            this->m_data = other.m_data;

            *this = std::move(other);
            new(&other) CompressedBitmap();
        }

    private:

        CompressedBitmap &operator=(const CompressedBitmap &other) = default;

    public:
        CompressedBitmap &operator=(CompressedBitmap &&other) {
            *this = other;
            new(&other) CompressedBitmap();

            return (*this);
        }

        ~CompressedBitmap() {
            Free();
        }

        DeviceObjectType DeviceObject() const {
            assert(m_size > 0);
            return dev::CompressedBitmap(m_data, m_size, m_positive_count);
        }

        void ResetAsync(cudaStream_t stream) {
            assert(m_size > 0);
            GROUTE_CUDA_CHECK(cudaMemsetAsync(m_positive_count, 0, sizeof(size_t), stream));
            GROUTE_CUDA_CHECK(cudaMemsetAsync(m_data, 0, sizeof(uint64_t) * get_num_words(), stream));
        }

        void ResetAsync(const groute::Stream &stream) {
            ResetAsync(stream.cuda_stream);
        }

        void SetBitAsync(index_t pos, cudaStream_t stream) {
            dev::kernels::SetBit << < 1, 1, 0, stream >> > (this->DeviceObject(), pos);
        }

        void SetBitAsync(index_t pos, const groute::Stream &stream) {
            SetBitAsync(pos, stream.cuda_stream);
        }


        void SetBitsAsync(cudaStream_t stream, const groute::dev::WorkSourceArray<index_t> &work_source) {
            dim3 grid_dims, block_dims;
            KernelSizing(grid_dims, block_dims, work_source.get_size());
            dev::kernels::SetBits << < grid_dims, block_dims, 0, stream >> > (this->DeviceObject(), work_source);
        }

        void SetBitsAsync(const groute::Stream &stream, const groute::dev::WorkSourceArray<index_t> &work_source) {
            SetBitsAsync(stream.cuda_stream, work_source);
        }

        void Swap(CompressedBitmap &other) {
            std::swap(m_data, other.m_data);
            std::swap(m_size, other.m_size);
            std::swap(m_positive_count, other.m_positive_count);
            std::swap(m_host_positive_count, other.m_host_positive_count);
        }

        size_t GetSize() const {
            return m_size;
        }

        size_t GetPositiveCount(const groute::Stream &stream) {
            GROUTE_CUDA_CHECK(cudaMemcpyAsync(m_host_positive_count,
                                              m_positive_count,
                                              sizeof(size_t),
                                              cudaMemcpyDeviceToHost,
                                              stream.cuda_stream));
            stream.Sync();

            assert(*m_host_positive_count <= m_size);
            return *m_host_positive_count;
        }

    private:
        uint64_t *m_data;
        size_t m_size;
        size_t *m_positive_count;
        size_t *m_host_positive_count;

        static const uint64_t kBitsPerWord = 64;

        uint64_t get_num_words() const {
            return (m_size + kBitsPerWord - 1) / kBitsPerWord;
        }

        void Alloc() {
            if (m_size == 0)
                return;
            GROUTE_CUDA_CHECK(cudaMalloc(&m_data, sizeof(uint64_t) * get_num_words()));
            GROUTE_CUDA_CHECK(cudaMalloc(&m_positive_count, sizeof(size_t)));
            GROUTE_CUDA_CHECK(cudaMemset(m_positive_count, 0, sizeof(size_t)));
            GROUTE_CUDA_CHECK(cudaMallocHost(&m_host_positive_count, sizeof(size_t)));
        }

        void Free() {
            GROUTE_CUDA_CHECK(cudaFree(m_data));
            GROUTE_CUDA_CHECK(cudaFree(m_positive_count));
            GROUTE_CUDA_CHECK(cudaFreeHost(m_host_positive_count));
            m_data = nullptr;
            m_positive_count = nullptr;
            m_host_positive_count = nullptr;
            m_size = 0;
        }
    };
}

#endif //HYBRID_COMPRESSED_BITMAP_H
