#ifndef DEBUG_H
#define DEBUG_H

#ifdef __cplusplus
#include <cstdio>
#include <cassert>
#else
#include <stdio.h>
#include <assert.h>
#endif

#define dfprint(expr) (printf(#expr " = %.15g\n", expr))
#define dprint(expr) (printf(#expr " = %d\n", expr))
#define check(e) do {							\
    if ((e) != cudaSuccess) {						\
	fprintf(stderr, "%s\n", cudaGetErrorString(e));			\
	assert(0);							\
    }									\
    } while (0)

#endif	/* DEBUG_H */
