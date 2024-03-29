# HyTGraph: GPU-Accelerated Graph Processing with Hybrid Transfer Management #

[Link to the paper](https://arxiv.org/abs/2208.14935)
## 1. Introduction ##
This repo contains all the source code to build HyTGraph.

## 2. Installation ##

#### 2.1 Software Requirements ####
* CUDA == 10.x
* GCC == 5.x.0
* CMake >= 3.14
* Linux/Unix

#### 2.2 Hardware Requirements ####

* Intel/AMD X64 CPUs
* 32GB RAM (at least)
* NVIDIA GTX 1080 or Tesla P100 or NVIDIA 2080ti
* 50GB Storage space (at least)

### 2.3 Setup ###
1. Download

    git clone --recursive https://github.com/iDC-NEU/HyTGraph.git
    
2. Build

  - cd HyTGraph
  - mkdir build && cd build
  - cmake .. 
  - make -j 8

## 3. Dataset ##
We use a custom format for the datasets.

```
source_id  dst_id  edge_weight
```

The weight is optional, and the system will replace it with src% 64. Vertex id are separated by \t.

Note: The number of vertices in the dataset cannot be less than the number of segments, which may cause some segments to be empty. This cannot happen to real-world dataset, because it usually has hundreds of millions of vertices.

## 4. How to Run ##

```
./hybrid_cc -graphfile example.el -format market_big
        -format | input file format
        
./hybrid_sssp -graphfile example.el -format market_big -source_node 0 -weight_num 1      or
./hybrid_sssp -graphfile example_weight.el -format market_big -source_node 0 -weight_num 0
        -format | input file format 
        -source | SSSP root (unused when i > 1)
        -weight_num | 0:read weight from given el file 1:generate edge weight with (#src % 64) and need to use a file without weight
```

More parameter descriptions are provided in the file app_skeleton.h
## 5. Contact ##

For the technical questions, please contact: **Xin Ai** (aixin0@stumail.neu.edu.cn)

For the questions about the paper, please contact: **Qiange Wang** (wangqiange@stumail.neu.edu.cn)
