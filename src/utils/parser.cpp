/*
* Copyright 1997, Regents of the University of Minnesota
*
* Extracted from Metis io.c http://glaros.dtc.umn.edu/gkhome/metis/metis/download
*
* This file contains routines related to I/O
*
* Started 8/28/94
* George
*
* $Id: io.c 11932 2012-05-10 18:18:23Z dominique $
*
*/

#include <utils/parser.h>


//#include <crtdefs.h>
#include <iostream>
#include <fstream>
#include <sstream> 
#include <cstddef>
#include <cstdarg>
#include <cstring>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <set>
#include <vector>


struct Edge{
    idx_t src;
    idx_t dst;
};

struct EdgeWeighted{
    idx_t src;
    idx_t dst;
    idx_t weight;
};


/*************************************************************************/
/*! This function initializes a graph_t data structure */
/*************************************************************************/
void InitGraph(graph_t *graph)
{
    memset((void *) graph, 0, sizeof(graph_t));

    /* graph size constants */
    graph->nvtxs = -1;
    graph->nedges = -1;
    graph->ncon = -1;
    graph->mincut = -1;
    graph->minvol = -1;
    graph->nbnd = -1;

    /* memory for the graph structure */
    graph->xadj = NULL;
    graph->vwgt = NULL;
    graph->vsize = NULL;
    graph->adjncy = NULL;
    graph->adjwgt = NULL;
    graph->label = NULL;
    graph->cmap = NULL;
    graph->tvwgt = NULL;
    graph->invtvwgt = NULL;

    graph->readvw = false;
    graph->readew = false;

    /* by default these are set to true, but the can be explicitly changed afterwards */
    graph->free_xadj = 1;
    graph->free_vwgt = 1;
    graph->free_vsize = 1;
    graph->free_adjncy = 1;
    graph->free_adjwgt = 1;


    /* memory for the partition/refinement structure */
    graph->where = NULL;
    graph->pwgts = NULL;
    graph->id = NULL;
    graph->ed = NULL;
    graph->bndptr = NULL;
    graph->bndind = NULL;
    graph->nrinfo = NULL;
    graph->ckrinfo = NULL;
    graph->vkrinfo = NULL;

    /* linked-list structure */
    graph->coarser = NULL;
    graph->finer = NULL;
}

/*************************************************************************/
/*! This function creates and initializes a graph_t data structure */
/*************************************************************************/
graph_t *CreateGraph(void)
{
    graph_t *graph;

    graph = (graph_t *) malloc(sizeof(graph_t));

    InitGraph(graph);

    return graph;
}

/*************************************************************************/
/*! This function deallocates any memory stored in a graph */
/*************************************************************************/
void FreeGraph(graph_t **r_graph)
{
    graph_t *graph;

    graph = *r_graph;

    /* free graph structure */
    if (graph->free_xadj)
        free((void *) graph->xadj);
    if (graph->free_vwgt)
        free((void *) graph->vwgt);
    if (graph->free_vsize)
        free((void *) graph->vsize);
    if (graph->free_adjncy)
        free((void *) graph->adjncy);
    if (graph->free_adjwgt)
        free((void *) graph->adjwgt);

    /* free partition/refinement structure */
    //FreeRData(graph);

    free((void *) graph->tvwgt);
    free((void *) graph->invtvwgt);
    free((void *) graph->label);
    free((void *) graph->cmap);
    free((void *) graph);

    *r_graph = NULL;
}

//static int exit_on_error = 1;

/*************************************************************************/
/*! This function prints an error message and exits
*/
/*************************************************************************/
void errexit(const char *f_str, ...)
{
    va_list argp;

    va_start(argp, f_str);
    vfprintf(stderr, f_str, argp);
    va_end(argp);

    if (strlen(f_str) == 0 || f_str[strlen(f_str) - 1] != '\n')
        fprintf(stderr, "\n");
    fflush(stderr);

    if (/*exit_on_error*/ 1)
        exit(-2);

    /* abort(); */
}

/*************************************************************************
* This function opens a file
**************************************************************************/
FILE *gk_fopen(const char *fname, const char *mode, const char *msg)
{
    FILE *fp;
    char errmsg[8192];

    fp = fopen(fname, mode);
    if (fp != NULL)
        return fp;

    sprintf(errmsg, "file: %s, mode: %s, [%s]", fname, mode, msg);
    perror(errmsg);
    errexit("Failed on gk_fopen()\n");

    return NULL;
}


/*************************************************************************
* This function closes a file
**************************************************************************/
void gk_fclose(FILE *fp)
{
    fclose(fp);
}


/*************************************************************************/
/*! This function is the GKlib implementation of glibc's getline()
function.
\returns -1 if the EOF has been reached, otherwise it returns the
number of bytes read.
*/
/*************************************************************************/
ptrdiff_t gk_getline(char **lineptr, size_t *n, FILE *stream)
{
#ifdef HAVE_GETLINE
    return getline(lineptr, n, stream);
#else
    size_t i;
    int ch;

    if (feof(stream))
        return -1;

    /* Initial memory allocation if *lineptr is NULL */
    if (*lineptr == NULL || *n == 0)
    {
        *n = 1024;
        *lineptr = (char *) malloc((*n) * sizeof(char));
    }

    /* get into the main loop */
    i = 0;
    while ((ch = getc(stream)) != EOF)
    {
        (*lineptr)[i++] = (char) ch;

        /* reallocate memory if reached at the end of the buffer. The +1 is for '\0' */
        if (i + 1 == *n)
        {
            *n = 2 * (*n);
            *lineptr = (char *) realloc(*lineptr, (*n) * sizeof(char));
        }

        if (ch == '\n')
            break;
    }
    (*lineptr)[i] = '\0';

    return (i == 0 ? -1 : i);
#endif
}

/*************************************************************************/
/*! This function reads in a sparse graph */
/*************************************************************************/
graph_t *ReadGraph(char *filename)
{
    idx_t i, j, k, l, fmt, ncon, nfields, readew, readvw, readvs, edge, ewgt;
    idx_t *xadj, *adjncy, *vwgt, *adjwgt, *vsize;
    char *line = NULL, fmtstr[256], *curstr, *newstr;
    size_t lnlen = 0;
    FILE *fpin;
    graph_t *graph;

    graph = CreateGraph();


    return graph;
}

#ifdef WIN32
// Windows "host" byte order is little endian
static inline uint64_t le64toh(uint64_t x) {
    return x;
}

#endif

/*************************************************************************/
/*! This function reads in a sparse graph */
/*************************************************************************/
graph_t *ReadGraphGR(char *filename)
{
    idx_t *xadj, *adjncy, *vwgt, *adjwgt, *vsize;
    FILE *fpin;
    graph_t *graph;

    graph = CreateGraph();
    return graph;
}

graph_t *ReadGraphMarket(char *filename)
{
    idx_t *xadj, *adjncy, *vwgt, *adjwgt, *vsize;
    FILE *fpin;
    char *line = NULL;
    size_t lnlen = 0;
    graph_t *graph;

    graph = CreateGraph();


    return graph;
}


graph_t *ReadGraphMarket_bigdata(char *filename,idx_t weight_num)
{
    uint64_t *xadj;
    idx_t *adjncy, *vwgt, *adjwgt, *vsize;

    std::ifstream infile;
    infile.open(filename);
    std::stringstream ss;
    std::string line;

    graph_t *graph;

    graph = CreateGraph();

    int weighted = 0; // 0: no weight; 1: int weight; 2: float weight

    if(weight_num == 1){
        weighted = 2;
    }

    graph->nedges = 0;
    graph->nvtxs = 0;

    std::vector<uint64_t> xadj_pri;
    
    xadj_pri.resize(1);

    uint32_t src,dst;
    while(getline( infile, line )){
        ss.str("");
        ss.clear();
        ss << line;
                
        ss >> src;
        ss >> dst;

        if(graph->nvtxs < src)
            graph->nvtxs = src;
        if(graph->nvtxs < dst)
            graph->nvtxs = dst;

        graph->nedges++;

        if(xadj_pri.size() <= src)
        {
            xadj_pri.resize(src * 2);
        }
        xadj_pri[src]++;

    }
    infile.close();
    graph->nvtxs++;

    //vwgt = graph->vwgt = (idx_t *) calloc((0 * graph->nvtxs), sizeof(idx_t));  // file doesn't store node weights though.
    graph->readvw = false;

    xadj = graph->xadj = (uint64_t *) calloc((graph->nvtxs + 1), sizeof(uint64_t));
    if(weighted == 0){
        adjncy = graph->adjncy = (idx_t *) calloc((graph->nedges), sizeof(uint32_t));
    }
    else{
        graph->readew = true;
        adjncy = graph->adjncy = (idx_t *) calloc((graph->nedges), sizeof(uint32_t));
        adjwgt = graph->adjwgt = (idx_t *) calloc((graph->nedges), sizeof(uint32_t));
    }

    idx_t edge_idx = 0;

    uint64_t count = 0;
    for (idx_t src = 0; src < graph->nvtxs; src++)
    {
        xadj[src] = count;
        count += xadj_pri[src];
    }
    xadj[graph->nvtxs] = graph->nedges;
    

    infile.open(filename);

    idx_t *outDegreeCounter  = (idx_t *) calloc((graph->nvtxs + 1), sizeof(idx_t));
    for(idx_t i=0; i<graph->nvtxs; i++)
        outDegreeCounter[i] = 0;
    uint32_t weight;
    while(getline( infile, line )){
        
        ss.str("");
        ss.clear();
        ss << line;
                
        ss >> src;
        ss >> dst;

 
        uint64_t location = xadj[src] + outDegreeCounter[src];                
        adjncy[location] = dst;
        outDegreeCounter[src]++; 

        if(weighted == 2){
            adjwgt[location] = src % 64;
        }
        else if(weighted == 1){
            ss >> weight;
            adjwgt[location] = weight;
        }

    }
    infile.close();
    delete[] outDegreeCounter;

    return graph;
}

