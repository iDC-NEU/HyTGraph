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
#ifndef APP_SKELETON_H
#define APP_SKELETON_H

#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>

#include <gflags/gflags.h>
#include <utils/utils.h>
#include <utils/interactor.h>
#include <groute/internal/cuda_utils.h>

// App
DEFINE_bool(interactive, false, "Run an interactive session");
DEFINE_string(cmdfile, "", "A file with commands to execute");
DEFINE_int32(max_iteration, 50, "Maximum iterated rounds");
DEFINE_bool(priority, false, "Enable priority scheduling (may not available for all algorithms)");
DEFINE_int32(prio_delta, 0, "The soft priority delta");

// DO Algorithms
DEFINE_double(edge_factor, 1.0, "Unvisited edges = edge_factor * nedges");
DEFINE_double(alpha, 0.8, "DO BFS/SSSP/BC heuristic parameter");
DEFINE_double(beta, 0.4, "DO BFS/SSSP/BC heuristic parameter");

// General
DEFINE_string(output, "", "File to store output to");
DEFINE_bool(check, false, "Check results");
DEFINE_bool(verbose, true, "Verbose prints");
DEFINE_bool(trace, false, "Trace prints (effects performance)");
DEFINE_string(lb_push, "", "load-balancing policy for push (none/coarse/fine/hybrid)");
DEFINE_string(lb_pull, "", "load-balancing policy for pull (none/coarse/fine/hybrid)");

// Input file/format
DEFINE_string(graphfile, "", "A file with a graph in Dimacs 10 format");
DEFINE_string(format, "gr", "graph format (gr/market/metis formats are supported)");
DEFINE_bool(undirected, false, "treat input graph as undirected graph");
DEFINE_string(json, "", "output running data as json format");
DEFINE_bool(stats, false, "Print graph statistics");

// Graph generation
DEFINE_bool(gen_graph, false, "Generate a random graph");
DEFINE_int32(gen_nnodes, 100000, "Number of nodes for random graph generation");
DEFINE_int32(gen_factor, 10, "A factor number for graph generation");
DEFINE_int32(gen_method, 0,
             "Select the requested graph generation method: \n\t0: Random graph \n\t1: Two-way chain graph without segment intersection \n\t2: Two-way chain graph with intersection \n\t3: Full cliques per device without segment intersection");
DEFINE_bool(gen_weights, false, "Generate edge weights if missing in graph input");
DEFINE_int32(gen_weight_range, 100,
             "The range to generate edge weights from (coordinate this parameter with nf-delta if running sssp-nf)");


// System
DEFINE_int32(block_size, 256, "Block size for traversal kernels");
DEFINE_int32(grid_size, 20, "Grid size for traversal kernels");


// Worklist
DEFINE_double(wl_alloc_factor, 0.2, "Worklist allocation factor");
DEFINE_bool(wl_sort, false, "Sort worklist by node id");
DEFINE_bool(wl_unique, false, "Unique duplicated nodes");

//byaixin
DEFINE_int32(n_stream, 4, "the number of stream");
DEFINE_int32(SEGMENT, 20, "the number of segment");
DEFINE_int32(priority_a, 1, "priority");
DEFINE_int32(hybrid, 2, "0:zerocopy 1:explicit 2：hybrid");
DEFINE_int32(residence, 1, "residence");
DEFINE_int32(weight_num, 0, "1:all weight = 1");


template<typename App>
struct Skeleton {
    int operator()(int argc, char **argv) {
        gflags::ParseCommandLineFlags(&argc, &argv, true);
        int exit = 0;

        if (!FLAGS_cmdfile.empty()) {
            FileInteractor file_interactor(FLAGS_cmdfile);
            std::cout << std::endl << "Starting a command file " << App::Name() << " session" << std::endl;
            RunInteractor(file_interactor);
        } else if (FLAGS_interactive) {
            ConsoleInteractor console_interactor;
            std::cout << std::endl << "Starting an interactive " << App::Name() << " session" << std::endl;
            RunInteractor(console_interactor);
        } else {
            NoInteractor no_interactor;
            RunInteractor(no_interactor);
        }

        App::Cleanup();

        return exit;
    }

    int RunInteractor(IInteractor &interactor) {
        int exit = 0;

        if (interactor.RunFirst()) exit = Run(); // run the first round

        while (true) {
            gflags::FlagSaver fs; // This saves the flags state and restores all values on destruction
            std::string cmd;

            if (!interactor.GetNextCommand(cmd)) break;
            cmd.insert(0, " ");
            cmd.insert(0, App::Name()); // insert any string to emulate the process name usually passed on argv

            int argc;
            char **argv;
            stringToArgcArgv(cmd, &argc, &argv);
            gflags::ParseCommandLineFlags(&argc, &argv, false);
            freeArgcArgv(&argc, &argv);

            exit = Run();
        }

        return exit;
    }

    int Run() {

        int num_actual_gpus;

        GROUTE_CUDA_CHECK(cudaGetDeviceCount(&num_actual_gpus));

        bool overall = true;

        printf("\nTesting single GPU %s\n", App::NameUpper());
        printf("--------------------\n\n");

        overall &= App::Single();


        printf("Overall: Test %s\n", overall ? "passed" : "FAILED");
        return 0;
    }
};


#endif // APP_SKELETON_H
