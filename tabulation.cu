//                                                                   -*- C++ -*-

#include "tabulation.cuh"
#include <cstdio>

#include "debug.h"

static cudaChannelFormatDesc fdesc = {
    32, 0, 0, 0, cudaChannelFormatKindFloat
};

#define VALUE_INDEX(min, max, n, i)		\
    ((min) + (real) (i) / (real) (n) * ((max) - (min)))

tabulation_t tabulation(func f, real min, real max, int n)
{
    float *data;
    cudaArray_t array;
    int i;
    tabulation_t t = { 0, min, max, n };
    struct cudaResourceDesc res;
    struct cudaTextureDesc tex;
    cudaError_t err;

    data = (float *) malloc((n+1) * sizeof(float));

    for (i = 0; i < n; i++)
	data[i] = (float) f(VALUE_INDEX(min, max, n, i));

    err = cudaMallocArray(&array, &fdesc, n+1);
    check(err);
    err = cudaMemcpyToArray(array, 0, 0, data, (n+1) * sizeof(float),
			    cudaMemcpyHostToDevice);
    check(err);

    memset(&res, 0, sizeof(res));

    res.resType = cudaResourceTypeArray;
    res.res.array.array = array;

    memset(&tex, 0, sizeof(tex));

    tex.addressMode[0] = cudaAddressModeClamp;
    tex.filterMode = cudaFilterModeLinear;
    //tex.filterMode = cudaFilterModePoint;
    tex.readMode = cudaReadModeElementType;
    tex.normalizedCoords = 1;

    err = cudaCreateTextureObject(&(t.tex), &res, &tex, NULL);
    check(err);

    return t;
}

void retabulate(tabulation_t *t, func f, float min, float max, int n)
{
    struct cudaResourceDesc res;
    cudaError_t err;
        
    err = cudaGetTextureObjectResourceDesc(&res, t->tex);
    check(err);
    assert(res.resType == cudaResourceTypeArray);

    err = cudaFreeArray(res.res.array.array);
    check(err);

    err = cudaDestroyTextureObject(t->tex);
    check(err);

    *t = tabulation(f, min, max, n);
}


__device__ real tab_value(tabulation_t t, real index)
{
    return (real) tex1D<float>(t.tex, index);
}

__device__ real tab_get_safe(tabulation_t t, real x)
{
    assert(TAB_WITHIN(t, x));
    real index = TAB_INDEX(t, x);
    return TAB_VALUE(t, index);
}

#undef VALUE_INDEX
