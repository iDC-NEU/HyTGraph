# HyTGraph: GPU-Accelerated Graph Processing with Hybrid Transfer Management #

## 1. Introduction ##
This repo contains all the source code to build SEP-Graph++.

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

    git clone --recursive https://github.com/AiX-im/HyTGraph.git
    
2. Build

  - cd SEP-GraphPP
  - mkdir build && cd build
  - cmake .. 
  - make -j 8

## 3. Contact ##

For the technical questions, please contact: aixin0@stumail.neu.edu.cn
For the questions about the paper, please contact: wangqiange@stumail.neu.edu.cn
