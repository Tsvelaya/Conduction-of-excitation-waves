#ifndef STATEVAR_H
#define STATEVAR_H

#include "value.h"
#include "util.h"

#define DIMX 1024
#define DIMY 1024

#define NUM_ELEMENTS (DIMX * DIMY)

#define ADDRESS(i, j) (DIMX * (i) + (j))

#define GET_ARRAY(a, i, j) ((a)[ADDRESS((i), (j))])
#define SET_ARRAY(a, i, j, x) ((a)[ADDRESS((i), (j))] = (x))
#define GET_ARRAY_ADDRESS(a, idx) ((a)[(idx)])
#define SET_ARRAY_ADDRESS(a, idx, x) ((a)[(idx)] = (x))

/* #define GET_VOLT(volt, i, j) (volt[ADDRESS((i), (j))]) */
/* #define SET_VOLT(volt, i, j, value) (volt[ADDRESS((i), (j))] = (value)) */
/* #define GET_VOLT_ADDRESS(volt, address) (volt[(address)]) */
/* #define SET_VOLT_ADDRESS(volt, address, value) (volt[(address)] = (value)) */

typedef real *voltage_t;

#define GET_VOLT GET_ARRAY
#define SET_VOLT SET_ARRAY
#define GET_VOLT_ADDRESS GET_ARRAY_ADDRESS
#define SET_VOLT_ADDRESS SET_ARRAY_ADDRESS

voltage_t create_voltage(location_t loc);
void destroy_voltage(voltage_t v, location_t loc);
void copy_voltage(voltage_t dst, voltage_t src, direction_t dir);

typedef struct {
    real *Ca, *Ca_JSR, *Ca_NSR;            /* [Ca] */
    real *M, *H, *J;                           /* INa */
    real *Y;                       /* If  */
    real *Xr;                              /* IKr */
    real *Xs1, *Xs2;                           /* IKs */
    real *R, *S, *Sslow;                       /* Ito */
    real *D, *F, *FCa;                 /* ICaL */
    real *B, *G;                   /* ICaT */
    real *P_o1;
} var_array_t;

typedef struct {
    real *next_i, *prev_i, *next_j, *prev_j;
    real *dxy_top_right, *dxy_top_left, *dxy_bottom_right, *dxy_bottom_left;
    real *dyx_top_right, *dyx_top_left, *dyx_bottom_right, *dyx_bottom_left;
    /*i -> bottom, j -> right*/
    /*dxx_next_j == next_j, dyy_next_i == next_i*/
} weights_array_t;

#define GET_STRUCT(var_array, member, i, j)	\
    (var_array->member[ADDRESS(i, j)])
#define SET_STRUCT(var_array, member, i, j, value)		\
    (var_array->member[ADDRESS(i, j)] = (value))

#define GET_STRUCT_ADDRESS(var_array, member, address)	\
    (var_array->member[address])
#define SET_STRUCT_ADDRESS(var_array, member, address, value)	\
    (var_array->member[address] = (value))

#define create_struct(loc, type) ((type *) create_struct_(loc, sizeof(type)))
#define destroy_struct(p, loc) (destroy_struct_(p, loc, sizeof(*p)))
#define copy_struct(dst, src, dir) (copy_struct_(dst, src, dir, sizeof(*dst)))

void set_uniform_weights(weights_array_t *weights_dev);
void set_block_weights(weights_array_t *weights_dev);
void set_half_anisotropy(weights_array_t *weights_dev, float ratio);

/* -- Implementation -- */

void *create_struct_(location_t loc, size_t size);
void destroy_struct_(void *v, location_t loc, size_t size);
void copy_struct_(void *dst, void *src, direction_t dir, size_t size);

#endif	/* STATEVAR_H */
