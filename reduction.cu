//                                                                -*- C++ -*-

/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


#include "reduction.h"
#include "util.h"

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<typename T>
struct SharedMemory
{
    __device__ inline operator       T *() {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

// specialize for double to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<double>
{
    __device__ inline operator       double *() {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }

    __device__ inline operator const double *() const {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }
};


/*  Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <typename T, unsigned int blockSize, bool nIsPow2>
__global__ void reduce(T *inp, T *out, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
    unsigned int gridSize = blockSize * 2 *gridDim.x;

    T sum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n) {
        sum += inp[i];

        // ensure we don't read out of bounds
	// this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
            sum += inp[i+blockSize];

        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = sum;
    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 512) {
        if (tid < 256)
            sdata[tid] = sum = sum + sdata[tid + 256];
        __syncthreads();
    }

    if (blockSize >= 256) {
        if (tid < 128)
            sdata[tid] = sum = sum + sdata[tid + 128];
        __syncthreads();
    }

    if (blockSize >= 128) {
        if (tid <  64)
            sdata[tid] = sum = sum + sdata[tid +  64];
        __syncthreads();
    }

    if (tid < 32) {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile T *smem = sdata;

        if (blockSize >=  64)
            smem[tid] = sum = sum + smem[tid + 32];

        if (blockSize >=  32)
            smem[tid] = sum = sum + smem[tid + 16];

        if (blockSize >=  16)
            smem[tid] = sum = sum + smem[tid +  8];

        if (blockSize >=   8)
            smem[tid] = sum = sum + smem[tid +  4];

        if (blockSize >=   4)
            smem[tid] = sum = sum + smem[tid +  2];

        if (blockSize >=   2)
            smem[tid] = sum = sum + smem[tid +  1];
    }

    // write result for this block to global mem
    if (tid == 0)
        out[blockIdx.x] = sdata[0];
}


bool is_pow2(unsigned int x)
{
    return (x & (x-1)) == 0;
}

unsigned int next_pow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

template <typename T>
void part_reduce(int n, int threads, int blocks, T *inp, T *out)
{
    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smem = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

    if (is_pow2(n)) {
	switch (threads) {
	    case 512:
		reduce<T, 512, true><<< blocks, threads, smem >>>(inp, out, n); break;
	    case 256:
		reduce<T, 256, true><<< blocks, threads, smem >>>(inp, out, n); break;
	    case 128:
		reduce<T, 128, true><<< blocks, threads, smem >>>(inp, out, n); break;
	    case 64:
		reduce<T,  64, true><<< blocks, threads, smem >>>(inp, out, n); break;
	    case 32:
		reduce<T,  32, true><<< blocks, threads, smem >>>(inp, out, n); break;
	    case 16:
		reduce<T,  16, true><<< blocks, threads, smem >>>(inp, out, n); break;
	    case  8:
		reduce<T,   8, true><<< blocks, threads, smem >>>(inp, out, n); break;
	    case  4:
		reduce<T,   4, true><<< blocks, threads, smem >>>(inp, out, n); break;
	    case  2:
		reduce<T,   2, true><<< blocks, threads, smem >>>(inp, out, n); break;
	    case  1:
		reduce<T,   1, true><<< blocks, threads, smem >>>(inp, out, n); break;
	}
    } else {
	switch (threads)
	    {
	    case 512:
		reduce<T, 512, false><<< blocks, threads, smem >>>(inp, out, n); break;
	    case 256:
		reduce<T, 256, false><<< blocks, threads, smem >>>(inp, out, n); break;
	    case 128:
		reduce<T, 128, false><<< blocks, threads, smem >>>(inp, out, n); break;
	    case 64:
		reduce<T,  64, false><<< blocks, threads, smem >>>(inp, out, n); break;
	    case 32:
		reduce<T,  32, false><<< blocks, threads, smem >>>(inp, out, n); break;
	    case 16:
		reduce<T,  16, false><<< blocks, threads, smem >>>(inp, out, n); break;
	    case  8:
		reduce<T,   8, false><<< blocks, threads, smem >>>(inp, out, n); break;
	    case  4:
		reduce<T,   4, false><<< blocks, threads, smem >>>(inp, out, n); break;
	    case  2:
		reduce<T,   2, false><<< blocks, threads, smem >>>(inp, out, n); break;
	    case  1:
		reduce<T,   1, false><<< blocks, threads, smem >>>(inp, out, n); break;
	    }
    }
}

static inline int div_up(int a, int b)
{
    return (a + (b - 1)) / b;
}

void get_dims(int n, int &threads, int &blocks)
{
    threads = n < REDUCTION_THREADS * 2 ?
		  next_pow2(div_up(n, 2)) : REDUCTION_THREADS;
    blocks = div_up(n, 2 * threads);
}

#define REDUCE(type)							\
    extern "C" type sum_##type(int n, type *inp) {			\
        static int size = 0;						\
	static type *out = 0;						\
	type res;							\
	int threads, blocks;						\
									\
	get_dims(n, threads, blocks);					\
									\
	if (size < blocks) {						\
	    size = blocks;						\
	    if (!out)							\
		out = (type *)xmalloc(sizeof(type) * size, DEVICE);	\
	    else {							\
  	        xfree(out, DEVICE);					\
		out = (type *)xmalloc(sizeof(type) * size, DEVICE);	\
	    }								\
	}								\
									\
        part_reduce<type>(n, threads, blocks, inp, out);		\
	n = blocks;							\
									\
	while (n > 1) {							\
	    get_dims(n, threads, blocks);				\
	    part_reduce<type>(n, threads, blocks, out, out);		\
	    n = blocks;							\
	}								\
									\
        xmemcpy(&res, out, sizeof(type), DEVICE_TO_HOST);	        \
									\
	return res;							\
    }

REDUCE(int)
REDUCE(real)
