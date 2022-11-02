//                                                                   -*- C++ -*-

#include "statevar.h"
#include "cuda.h"
#include "util.h"
#include "activation.h"
#include "reduction.h"

#define ACT_THREADS 512
#define ACT_THRESH REAL(-60.0)

static actmap_t act = NULL;

extern "C"
actmap_t create_actmap(location_t loc)
{
    return (actmap_t) xmalloc(NUM_ELEMENTS * sizeof(int), loc);
}

__global__
void activation_kernel(voltage_t volt, int *act, char *mask, int step)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (volt[i] >= ACT_THRESH && mask[i]) {
	act[i] = step + 1;
	mask[i] = 0;
    }
}

extern "C"
actmap_t get_activation()
{
    static actmap_t res = NULL;

    if (!res) res = create_actmap(HOST);
    xmemcpy(res, act, sizeof(int) * NUM_ELEMENTS, DEVICE_TO_HOST);
    return res;
}

extern "C"
void activation(voltage_t volt, medium_t med, int step)
{
    static char *mask = NULL;

    if (!mask) {
	mask = (char *) xmalloc(NUM_ELEMENTS, DEVICE);
	if (med == NULL) cudaMemset(mask, 1, NUM_ELEMENTS);
	else xmemcpy(mask, med, NUM_ELEMENTS, HOST_TO_DEVICE);

	act  = create_actmap(DEVICE);
	cudaMemset(act, 0, sizeof(int) * NUM_ELEMENTS);
    }

    activation_kernel
    	<<< NUM_ELEMENTS / ACT_THREADS, ACT_THREADS, 0, Cuda_Stream[COMPUTATION] >>>
    	(volt, act, mask, step);

}

