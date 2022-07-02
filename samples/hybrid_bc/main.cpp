// ----------------------------------------------------------------
// SEP-Graph: Finding Shortest Execution Paths for Graph Processing under a Hybrid Framework on GPU
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE
// in the root directory of this source distribution.
// ----------------------------------------------------------------
#include <cstdio>
#include <cuda_runtime.h>
#include <gflags/gflags.h>
#include <iostream>
#include <utils/utils.h>
#include <utils/interactor.h>
#include <utils/app_skeleton.h>

bool HybridBetweennessCentrality();

bool HybridBC();

void CleanupGraphs();


namespace pr
{
    struct App
    {
        static const char *Name()
        { return "betweenness centrality"; }

        static const char *NameUpper()
        { return "Betweenness Centrality"; }

        static bool Single()
        {
//            HybridBetweennessCentrality();
            HybridBC();
        }

        static void Cleanup()
        { CleanupGraphs(); }
    };
}

int main(int argc, char **argv)
{
    Skeleton<pr::App> app;
    int exit = app(argc, argv);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return exit;
}
