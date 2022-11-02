//                                                                   -*- C++ -*-
#ifndef TABULATION_CUH
#define TABULATION_CUH

#include "value.h"

typedef struct {
    cudaTextureObject_t tex;
    real min, max;
    int size;
} tabulation_t;

typedef double (*func)(double);

tabulation_t tabulation(func f, real min, real max, int n);
void retabulate(tabulation_t *t, func f, float min, float max, int n);

#define TAB_WITHIN(t, x) ((x) >= (t).min && (x) <= (t).max)
#define TAB_INDEX(t, x)	(((x) - (t).min) / ((t).max - (t).min))
#define TAB_VALUE(t, i) (tex1D<float>(t.tex, (i)))

#ifndef NDEBUG
#define TAB_GET(t, x) (TAB_VALUE((t), TAB_INDEX((t), (x))))
#else
#define TAB_GET(t, i) tab_get_safe(t, i)
#endif

__device__ real tab_get_safe(tabulation_t t, real index);

#endif	/* TABULATION_CUH */
