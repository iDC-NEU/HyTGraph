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

#ifndef __GROUTE_CONTEXT_H
#define __GROUTE_CONTEXT_H

#include <initializer_list>
#include <vector>
#include <map>
#include <set>
#include <memory>
#include <future>
#include <functional>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>

#include <groute/internal/cuda_utils.h>
#include <groute/common.h>
#include <groute/event_pool.h>

namespace groute {

    struct ContextConfiguration
    {
        bool verbose, trace;

        ContextConfiguration() : verbose(true), trace(false) { }
    };

    /*
    * @brief The global groute context 
    */
    class Context
    {
        //
        // The context provides an abstraction layer between virtual 'endpoints' and the actual physical devices in the system.
        // In addition, it provides global services, such as memory-copy lanes for queing asynchronous copy operations, and event management.
        //

        std::map<Endpoint, device_t> m_endpoint_map; // Maps from endpoints to physical devices   

        std::set<int> m_physical_devs; // The physical devices currently in use by this context
        std::map<int, std::unique_ptr<EventPool> > m_event_pools;

        mutable std::mutex m_mutex;

        void InitPhysicalDevs()
        {
            for (auto& p : m_endpoint_map)
            {
                if (p.second == Device::Host) continue;
                m_physical_devs.insert(p.second);
            }

            for (int physical_dev_i : m_physical_devs)
            {
                GROUTE_CUDA_CHECK(cudaSetDevice(physical_dev_i));
                for (int physical_dev_j : m_physical_devs)
                    if (physical_dev_i != physical_dev_j)
                        cudaDeviceEnablePeerAccess(physical_dev_j, 0);
            }
        }

        void CreateEventPools()
        {            
            for (int physical_dev : m_physical_devs)
            {
                auto pool = make_unique<EventPool>(physical_dev);
                m_event_pools[physical_dev] = std::move(pool);
            }
        }

    public:
        ContextConfiguration configuration;

        Context()
        {
            int actual_ngpus;
            GROUTE_CUDA_CHECK(cudaGetDeviceCount(&actual_ngpus));

            // build a simple one-to-one endpoint map  
            for (int physical_dev = 0; physical_dev < actual_ngpus; ++physical_dev)
            {
                m_endpoint_map[physical_dev] = physical_dev;
            }
            // host
            m_endpoint_map[Endpoint::HostEndpoint(0)] = Device::Host;

            InitPhysicalDevs();
            CreateEventPools();
        }

        ~Context()
        {
            if (configuration.verbose)
            {
                printf("\nContext status:"); 
                PrintStatus(); 
                printf("\n");
            }
        }

        void CacheEvents(size_t per_endpoint)
        {
            for (int physical_dev : m_physical_devs)
            {
                int endpoints = 0; // Count the number of endpoints using the physical device  
                for (auto& p : m_endpoint_map)
                {
                    if (p.second == physical_dev) ++endpoints;
                }

                m_event_pools.at(physical_dev)->CacheEvents(per_endpoint * endpoints);
            }
        }

        const std::map<Endpoint, int>& GetEndpointMap() const { return m_endpoint_map; }

        int GetPhysicalDevice(Endpoint endpoint) const { return m_endpoint_map.at(endpoint); }

        void SetDevice(Endpoint endpoint) const
        {
            if (endpoint.IsHost()) return;

            int current_physical_dev, requested_physical_dev = m_endpoint_map.at(endpoint);
            GROUTE_CUDA_CHECK(cudaGetDevice(&current_physical_dev));

            if (current_physical_dev == requested_physical_dev) return;
            GROUTE_CUDA_CHECK(cudaSetDevice(requested_physical_dev));
        }

        void SyncDevice(Endpoint endpoint) const
        {
            if (endpoint.IsHost()) return;

            SetDevice(endpoint);
            GROUTE_CUDA_CHECK(cudaDeviceSynchronize());
        }

        void SyncAllDevices() const
        {
            for (int physical_dev_i : m_physical_devs)
            {
                GROUTE_CUDA_CHECK(cudaSetDevice(physical_dev_i));
                GROUTE_CUDA_CHECK(cudaDeviceSynchronize());
            }
        }

        Stream CreateStream(Endpoint endpoint, StreamPriority priority = SP_Default) const
        {
            return Stream(m_endpoint_map.at(endpoint), priority);
        }

        Stream CreateStream(StreamPriority priority = SP_Default) const
        {
            return Stream(priority);
        }

        EventPool& GetEventPool(Endpoint endpoint) const
        {
            return *m_event_pools.at(m_endpoint_map.at(endpoint));
        }

        Event RecordEvent(Endpoint endpoint, cudaStream_t stream) const
        {
            return m_event_pools.at(m_endpoint_map.at(endpoint))->Record(stream);
        }

        Event RecordEvent(cudaStream_t stream) const
        {
            int current_physical_dev;
            GROUTE_CUDA_CHECK(cudaGetDevice(&current_physical_dev));
            return m_event_pools.at(current_physical_dev)->Record(stream);
        }

        Event RecordEvent(Endpoint endpoint, const Stream& stream) const
        {
            return RecordEvent(endpoint, stream.cuda_stream);
        }

        Event RecordEvent(const Stream& stream) const
        {
            return RecordEvent(stream.cuda_stream);
        }


        void PrintStatus() const
        {
            printf("\n\tEndpoint map (virtual endpoint, physical device): ");
            for (auto& p : m_endpoint_map)
            {
                printf(" (%d, %d)", (Endpoint::identity_type)p.first, p.second);
            }

            printf("\n\tEvent pools (device, events): ");
            for (auto& p : m_event_pools)
            {
                printf(" (%d, %d)", p.first, p.second->GetCachedEventsNum());
            }
        }
    };
}

#endif // __GROUTE_CONTEXT_H
