//                                                                   -*- C++ -*-
#include "config.h"
#include "debug.h"
#include "util.h"

cudaStream_t Cuda_Stream[2];

extern int device_num;

extern "C"
void init_cuda(void)
{
    int i;
    cudaError_t e;
    int dev;

    cudaGetDeviceCount(&dev);
    if (device_num < dev)
	cudaSetDevice(device_num);

#if PRECISION == FLOAT
    e = cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
#elif PRECISION == DOUBLE
    e = cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
#endif
    check(e);

    for (i = 0; i < SIZE(Cuda_Stream); i++)
	cudaStreamCreate(&Cuda_Stream[i]);
}
