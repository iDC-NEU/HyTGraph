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

#ifndef __GROUTE_COMMON_H
#define __GROUTE_COMMON_H

#include <map>
#include <future>
#include <vector>
#include <climits>
#include <type_traits>
#include <assert.h>
#include <groute/internal/cuda_utils.h>


namespace {
    static inline __host__ __device__ size_t round_up(
            size_t numerator, size_t denominator) {
        return (numerator + denominator - 1) / denominator;
    }

    // Adapted from http://stackoverflow.com/questions/466204/rounding-up-to-nearest-power-of-2
    template<typename UnsignedType>
    UnsignedType next_power_2(UnsignedType v) {
        static_assert(std::is_unsigned<UnsignedType>::value, "Only works for unsigned types");
        --v;
        for (int i = 1; i < sizeof(v) * CHAR_BIT; i *= 2) {
            v |= v >> i;
        }
        return ++v;
    }

    inline std::vector<int> range(int count, int from = 0) {
        std::vector<int> vec(count);
        for (int i = 0; i < count; i++) {
            vec[i] = from + i;
        }

        return std::move(vec);
    }
}


namespace groute {

#ifdef WIN32
    using exception = std::exception;
#else

    class exception : public std::exception {
        const char *m_message;

    public:
        exception(const char *message) : m_message(message) {}

        exception() : m_message(nullptr) {}

        const char *what() const noexcept {
            return m_message == nullptr ? std::exception::what() : m_message;
        }
    };

#endif

    typedef int device_t;

    /**
    * @brief Device (physical) related metadata  
    */
    class Device {
    public:
        Device() = delete;

        enum : int {
            Null = INT32_MIN,
            Host = -1
        };
    };

    /**
    * @brief Represents an Endpoint (possibly virtual) in the system
    * @note By convention, Host endpoints should be represented by negative numbers and GPU endpoints by non-negative (i.e. >=0) numbers
    */
    struct Endpoint {
        typedef int identity_type;

    private:
        identity_type m_identity;

        enum : identity_type {
            Null = INT32_MIN,
            Host = -1
        };

    public:
        Endpoint(identity_type identity) : m_identity(identity) {} // Implicit conversion from int
        Endpoint() : m_identity(Null) {}

        explicit operator identity_type() const { return m_identity; } // Use this to obtain the identity value

        bool operator<(const Endpoint &other) const { return m_identity < other.m_identity; }

        bool operator<=(const Endpoint &other) const { return m_identity <= other.m_identity; }

        bool operator>(const Endpoint &other) const { return m_identity > other.m_identity; }

        bool operator>=(const Endpoint &other) const { return m_identity >= other.m_identity; }

        bool operator==(const Endpoint &other) const { return m_identity == other.m_identity; }

        bool IsGPU() const { return m_identity >= 0; } // Any non-negative number can represent a GPU endpoint
        bool IsHost() const { return m_identity <= Host && m_identity != Null; } // Any negative number but 'Null' can represent a Host endpoint
        bool IsNull() const { return m_identity == Null; }

        static std::vector<Endpoint> Range(int count, identity_type from = 0, bool reverse = false) {
            std::vector<Endpoint> vec(count);
            for (int i = 0; i < count; i++) {
                vec[i] = reverse ? from - i : from + i;
            }

            return std::move(vec);
        }

        static Endpoint HostEndpoint(int i) { return Host * (i + 1); }  // Get the i'th Host endpoint (-1, -2, ...)
        static Endpoint GPUEndpoint(int i) { return i; }           // Get the i'th GPU endpoint (0, 1, 2, ...)
    };

    enum StreamPriority {
        SP_Default, SP_High, SP_Low
    };

    class Stream {
    public:
        cudaStream_t cuda_stream;
        cudaEvent_t sync_event;

        Stream(int physical_dev, StreamPriority priority = SP_Default) : cuda_stream(nullptr), sync_event(nullptr) {
            GROUTE_CUDA_CHECK(cudaSetDevice(physical_dev));
            Init(priority);
        }

        Stream(StreamPriority priority) : cuda_stream(nullptr), sync_event(nullptr) {
            Init(priority);
        }

        Stream() : cuda_stream(nullptr), sync_event(nullptr) {
        }

        void Init(StreamPriority priority) {
            Destroy();

            if (priority == SP_Default) {
                GROUTE_CUDA_CHECK(cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking));
                GROUTE_CUDA_CHECK(cudaEventCreateWithFlags(&sync_event, cudaEventDisableTiming));
            } else {
                int leastPriority, greatestPriority;
                cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority); // range: [*greatestPriority, *leastPriority]

                GROUTE_CUDA_CHECK(
                        cudaStreamCreateWithPriority(&cuda_stream, cudaStreamNonBlocking, priority == SP_High ? greatestPriority : leastPriority));
                GROUTE_CUDA_CHECK(cudaEventCreateWithFlags(&sync_event, cudaEventDisableTiming));
            }
        }

        void Destroy() {
            if (cuda_stream != nullptr) GROUTE_CUDA_CHECK(cudaStreamDestroy(cuda_stream));
            if (sync_event != nullptr) GROUTE_CUDA_CHECK(cudaEventDestroy(sync_event));

            cuda_stream = nullptr;
            sync_event = nullptr;
        }

        Stream(const Stream &other) = delete;

        Stream(Stream &&other) : cuda_stream(other.cuda_stream), sync_event(other.sync_event) {
            other.cuda_stream = nullptr;
            other.sync_event = nullptr;
        }

        Stream &operator=(const Stream &other) = delete;

        Stream &operator=(Stream &&other) {
            Destroy();

            cuda_stream = other.cuda_stream;
            sync_event = other.sync_event;

            other.cuda_stream = nullptr;
            other.sync_event = nullptr;

            return *this;
        }

        ~Stream() {
            Destroy();
        }

        void Sync() const {
            GROUTE_CUDA_CHECK(cudaEventRecord(sync_event, cuda_stream));
            GROUTE_CUDA_CHECK(cudaEventSynchronize(sync_event));
        }

        void BeginSync() const {
            GROUTE_CUDA_CHECK(cudaEventRecord(sync_event, cuda_stream));
        }

        bool Query() const {
            return cudaEventQuery(sync_event) == cudaSuccess;
        }

        void EndSync() const {
            GROUTE_CUDA_CHECK(cudaEventSynchronize(sync_event));
        }
    };


    template<typename Future>
    bool is_ready(const Future &f) {
#ifdef WIN32
        return f._Is_ready();
#else
        return f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
#endif
    }

    template<typename T>
    std::shared_future<T> completed_future(const T &val) {
        std::promise<T> prom;
        std::shared_future<T> fut = prom.get_future();
        prom.set_value(val);
        return fut;
    }

    // workaround for VS C++11
    template<typename T, typename... Args>
    std::unique_ptr<T> make_unique(Args &&... args) {
        return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    }
}

namespace std {
    template<>
    struct hash<groute::Endpoint>
            : private hash<groute::Endpoint::identity_type> {    // hash functor for Endpoint (to enable usage as key)
        size_t operator()(const groute::Endpoint &endpoint) const {
            return hash<groute::Endpoint::identity_type>::operator()(
                    static_cast<groute::Endpoint::identity_type>(endpoint));
        }
    };
}

#endif // __GROUTE_COMMON_H
