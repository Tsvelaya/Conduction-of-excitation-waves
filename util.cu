//                                                                   -*- C++ -*-
#include <cassert>
#include <cstdio>

#include "util.h"
#include "debug.h"
#include "cuda.h"

extern "C"
void *xmalloc(size_t size, location_t location)
{
    void *p;
    cudaError_t e;

    switch (location) {
    case HOST:
        p = malloc(size);
        assert(p);
        return p;
    case DEVICE:
        e = cudaMalloc(&p, size);
        check(e);
        e = cudaMemset(p, 0, size);
        check(e);
        return p;
    case LOCKED:
	e = cudaMallocHost(&p, size);
	check(e);
	return p;
    default:
        assert(0);
    }

    return NULL;
}

extern "C"
void xfree(void *p, location_t location)
{
    switch (location) {
    case HOST:
        free(p);
        return;
    case DEVICE:
        cudaFree(p);
        return;
    default:
        assert(0);
    }
}

static cudaMemcpyKind kind_of_dir(direction_t dir)
{
    switch (dir) {
    case HOST_TO_HOST: return cudaMemcpyHostToHost;
    case HOST_TO_DEVICE: return cudaMemcpyHostToDevice;
    case DEVICE_TO_HOST: return cudaMemcpyDeviceToHost;
    case DEVICE_TO_DEVICE: return cudaMemcpyDeviceToDevice;
    default: assert(0);
    }

    return cudaMemcpyDefault; 			// Never get here
}

extern "C"
void xmemcpy(void *dst, void *src, size_t n, direction_t dir)
{
    cudaError_t e;
    if (dir == DEVICE_TO_HOST_ASYNC) {
	e = cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToHost,
			    Cuda_Stream[COPYING]);
	check(e);
	e = cudaStreamSynchronize(Cuda_Stream[COPYING]);
	check(e);
    } else {
	e = cudaMemcpy(dst, src, n, kind_of_dir(dir));
	check(e);
    }
}
