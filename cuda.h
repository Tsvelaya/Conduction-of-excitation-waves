#ifndef INIT_CUDA_H
#define INIT_CUDA_H

#ifdef __CUDACC__
extern cudaStream_t Cuda_Stream[2];
enum { COMPUTATION = 0, COPYING = 1 };
#endif

#ifdef __cplusplus
extern "C"
#endif
void init_cuda(void);

#endif	/* INIT_CUDA_H */
