cmake_minimum_required(VERSION 2.8)
add_definitions(-w)
project(SEP-Graph)
############## DEPENDENCIES ##############
find_package(CUDA 9 REQUIRED)

message("-- CUDA Version: ${CUDA_VERSION_STRING}")

# GFlags
add_subdirectory(deps/gflags)
include_directories(${PROJECT_BINARY_DIR}/deps/gflags/include)
link_directories(${PROJECT_BINARY_DIR}/deps/gflags/lib)

# CUB
include_directories(${PROJECT_SOURCE_DIR}/deps/cub)

# JSON
include_directories(${PROJECT_SOURCE_DIR}/deps/json)
############## BUILD ##############

set(EXTRA_LIBS gflags pthread ${CUDA_TOOLKIT_TARGET_DIR}/lib64/stubs/libcuda.so ${CUDA_TOOLKIT_TARGET_DIR}/lib64/libnvToolsExt.so)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("Debug mode")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_52,code=sm_52;-gencode;arch=compute_52,code=compute_52;-gencode;arch=compute_60,code=sm_60;-gencode;arch=compute_61,code=sm_61;-g;-lineinfo;-Xcompiler;-ggdb;-std=c++11 -O3)
else ()
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_52,code=sm_52;-gencode;arch=compute_52,code=compute_52;-gencode;arch=compute_60,code=sm_60;-gencode;arch=compute_61,code=sm_61;-O3;-DNDEBUG;-Xcompiler;-DNDEBUG;-std=c++11 -O3)
endif ()

set(CUDA_PROPAGATE_HOST_FLAGS OFF)

# Addresses a bug where code is not compiled as C++11 in non-CUDA code and older g++ versions
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -DNDEBUG")

message("-- Host Compiler: ${CUDA_HOST_COMPILER}")

include_directories(${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include)

# hybrid apps
cuda_add_executable(hybrid_pr src/utils/parser.cpp src/utils/utils.cpp src/utils/to_json.cpp #src/groute/graphs/csr_graph.cpp
       samples/hybrid_pr/hybrid_pr.cu
       samples/hybrid_pr/hybrid_pr_host.cpp
       samples/hybrid_pr/main.cpp)
target_link_libraries(hybrid_pr ${EXTRA_LIBS} pthread)

cuda_add_executable(hybrid_bfs src/utils/parser.cpp src/utils/utils.cpp src/utils/to_json.cpp #src/groute/graphs/csr_graph.cpp
        samples/hybrid_bfs/hybrid_bfs.cu
        samples/hybrid_bfs/hybrid_bfs_host.cpp
        samples/hybrid_bfs/main.cpp)
target_link_libraries(hybrid_bfs ${EXTRA_LIBS} pthread)
#$target_compile_definitions(hybrid_bfs PUBLIC ARRAY_BITMAP)

cuda_add_executable(hybrid_sssp src/utils/parser.cpp src/utils/utils.cpp src/utils/to_json.cpp #src/groute/graphs/csr_graph.cpp
        samples/hybrid_sssp/hybrid_sssp.cu
       samples/hybrid_sssp/hybrid_sssp_host.cpp
        samples/hybrid_sssp/main.cpp)
target_link_libraries(hybrid_sssp ${EXTRA_LIBS} pthread)

cuda_add_executable(hybrid_cc src/utils/parser.cpp src/utils/utils.cpp src/utils/to_json.cpp #src/groute/graphs/csr_graph.cpp
        samples/hybrid_cc/hybrid_cc.cu
        samples/hybrid_cc/main.cpp)
target_link_libraries(hybrid_cc ${EXTRA_LIBS} pthread)

add_executable(gtools src/utils/parser.cpp src/utils/utils.cpp
        samples/tools/gtools.cpp)
target_link_libraries(gtools gflags)
