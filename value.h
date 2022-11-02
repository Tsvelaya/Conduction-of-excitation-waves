#ifndef VALUE_H
#define VALUE_H

#include "config.h"

#if PRECISION == FLOAT
typedef float real;
#define REAL(a) (a##f)
#define sqrt sqrtf
#define exp expf
#define log logf
#elif PRECISION == DOUBLE
typedef double real;
#define REAL(a) (a)
#endif

#endif	/* VALUE_H */
