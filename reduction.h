#ifndef REDUCTION_H
#define REDUCTION_H

#include "value.h"
#define REDUCTION_THREADS 512

#ifdef __cplusplus
extern "C" {
#endif

int sum_int(int n, int *data);
real sum_real(int n, real *data);

#ifdef __cplusplus
}
#endif

#endif
