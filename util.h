#ifndef UTIL_H
#define UTIL_H

#ifdef __cplusplus
#include <cstdlib>
#else
#include <stdlib.h>
#endif

#define MIN(x,y) ((x) < (y) ? (x) : (y))
#define MAX(x,y) ((x) > (y) ? (x) : (y))
#define SQR(x) ((x) * (x))
#define CUBE(x) ((x) * (x) * (x))
#define SIZE(a) (sizeof(a) / sizeof(a[0]))

#define SWAP(a,b) do {				\
    typeof(a) tmp;				\
    tmp = a;					\
    a = b;					\
    b = tmp;					\
    } while (0)

typedef unsigned char uchar;

typedef enum { HOST = 1, DEVICE = 2, LOCKED = 3 } location_t;

typedef enum {
    HOST_TO_HOST = 0,
    HOST_TO_DEVICE = 1,
    DEVICE_TO_HOST = 2,
    DEVICE_TO_DEVICE = 3,
    DEVICE_TO_HOST_ASYNC
} direction_t;

#ifdef __cplusplus
extern "C" {
#endif

void *xmalloc(size_t size, location_t location);
void xfree(void *p, location_t location);
void xmemcpy(void *dst, void *src, size_t n, direction_t direction);
void sync_copy();

#ifdef __cplusplus
}
#endif

#endif	/* UTIL_H */
