#include <assert.h>
#include <string.h>

#include "value.h"
#include "statevar.h"
#include "parameters.cuh"

#define ALIGNMENT 4096
#define ALIGN(p) ( ((size_t)(p) + (ALIGNMENT - 1)) & (~(ALIGNMENT - 1)) )

void *create_struct_(location_t loc, size_t size)
{
    int k;
    void *v;
    real **m;

    v = xmalloc(size, HOST);
    m = (real **)v;

    for (k = 0; k < size / sizeof(real**); k++)
	m[k] = (real *)xmalloc(NUM_ELEMENTS * sizeof(real), loc);

    if (loc == DEVICE) {
	void *dev;

	dev = xmalloc(size, DEVICE);
	xmemcpy(dev, v, size, HOST_TO_DEVICE);

	xfree(v, HOST);
	v = dev;
    }

    return v;
}

void destroy_struct_(void *v, location_t loc, size_t size)
{
    int k;
    void *tmp = NULL;
    void *orig;
    real **m;

    orig = v;

    if (loc == DEVICE) {
        tmp = xmalloc(size, HOST);
	xmemcpy(tmp, v, size, DEVICE_TO_HOST);
	v = &tmp;
    }

    m = (real **)v;

    for (k = 0; k < size / sizeof(real **); k++)
	xfree(m[k], loc);

    xfree(orig, loc);
    if (tmp) xfree(tmp, HOST);
}

void copy_struct_(void *dst, void *src, direction_t dir, size_t size)
{
    int k;
    real **d, **s;
    void *vd = NULL, *vs = NULL;

    d = (real **) dst;
    s = (real **) src;

    switch (dir) {
    case HOST_TO_DEVICE:
        vd = xmalloc(size, HOST);
	xmemcpy(vd, dst, size, DEVICE_TO_HOST);
	d = (real **) vd;
	break;
    case DEVICE_TO_HOST:
        vs = xmalloc(size, HOST);
	xmemcpy(vs, src, size, DEVICE_TO_HOST);
	s = (real **) vs;
	break;
    case DEVICE_TO_DEVICE:
        vd = xmalloc(size, HOST);
	xmemcpy(vd, dst, size, DEVICE_TO_HOST);
	d = (real **) vd;
        vs = xmalloc(size, HOST);
	xmemcpy(vs, src, size, DEVICE_TO_HOST);
	s = (real **) vs;
	break;
    case HOST_TO_HOST:
	break;
    default:
	assert(0);
    }

    for (k = 0; k < size / sizeof(real **); k++)
	xmemcpy(d[k], s[k], NUM_ELEMENTS * sizeof(real), dir);

    if (vd) xfree(vd, HOST);
    if (vs) xfree(vs, HOST);
}



voltage_t create_voltage(location_t loc)
{
    voltage_t v;
    switch (loc) {
    case HOST: case LOCKED:
	return (voltage_t) xmalloc(NUM_ELEMENTS * sizeof(real), loc);
    case DEVICE:
        v = (real *) xmalloc((NUM_ELEMENTS + 2 * (DIMX + 1)) * sizeof(real)
                        + ALIGNMENT, DEVICE);
	v += DIMX + 1;
	v = (voltage_t) ALIGN(v);
	return v;
    }

    assert(0);
    return 0;
}

void destroy_voltage(voltage_t v, location_t loc)
{
    xfree(v, loc);
}


void copy_voltage(voltage_t dst, voltage_t src, direction_t dir)
{
    xmemcpy(dst, src, NUM_ELEMENTS * sizeof(real), dir);
}



void set_uniform_weights(weights_array_t *weights_dev)
{
    int i, j;
    weights_array_t *weights_host;

    weights_host = create_struct(HOST, weights_array_t);

    for (i = 0; i < DIMY; i++)
	for (j = 0; j < DIMX; j++) {
	    SET_STRUCT(weights_host, next_i, i, j, i == DIMY - 1 ? 0.0 : 1.0);
	    SET_STRUCT(weights_host, next_j, i, j, j == DIMX - 1 ? 0.0 : 1.0);
	    SET_STRUCT(weights_host, prev_i, i, j, i == 0 ? 0.0 : 1.0);
	    SET_STRUCT(weights_host, prev_j, i, j, j == 0 ? 0.0 : 1.0);
	}

    copy_struct(weights_dev, weights_host, HOST_TO_DEVICE);
    destroy_struct(weights_host, HOST);
}

void set_half_anisotropy(weights_array_t *weights_dev, float ratio)
{
    int i, j;
    weights_array_t *weights_host;

    weights_host = create_struct(HOST, weights_array_t);

    real dxx_1, dxx_2, dyy_1, dyy_2;
    real dxy_1 = 0.0, dyx_1 = 0.0, dxy_2 = 0.0, dyx_2 = 0.0;

    for (i = 0; i < DIMY; i++) {
        dxx_1 = 1/ratio/ratio;
        dxx_2 = 1.;
        dyy_2 = dxx_1;
        dyy_1 = dxx_2;
        
    	for (j = 0; j < EDGE; j++) {

    	    SET_STRUCT(weights_host, next_i, i, j, i == DIMY - 1 ? 0.0 : dyy_1 + 4*(j + 1)/DIMY); //for drift
    	    SET_STRUCT(weights_host, next_j, i, j, j == DIMX - 1 ? 0.0 : dxx_1 + 4*(j + 1)/DIMX); 
    	    SET_STRUCT(weights_host, prev_i, i, j, i == 0 ? 0.0 : dyy_1 + j/DIMY);
    	    SET_STRUCT(weights_host, prev_j, i, j, j == 0 ? 0.0 : dxx_1 + j/DIMX); 

    	   /* SET_STRUCT(weights_host, next_i, i, j, i == DIMY - 1 ? 0.0 : dyy_1);
    	    SET_STRUCT(weights_host, next_j, i, j, j == DIMX - 1 ? 0.0 : dxx_1);
    	    SET_STRUCT(weights_host, prev_i, i, j, i == 0 ? 0.0 : dyy_1);
    	    SET_STRUCT(weights_host, prev_j, i, j, j == 0 ? 0.0 : dxx_1); */

            SET_STRUCT(weights_host, dxy_top_left, i, j, (i == 0 || j == 0) ? 0.0 : dxy_1);
            SET_STRUCT(weights_host, dxy_top_right, i, j, (i == 0 || j == DIMX - 1) ? 0.0 : dxy_1);
            SET_STRUCT(weights_host, dxy_bottom_left, i, j, (i == DIMY - 1 || j == 0) ? 0.0 : dxy_1);
            SET_STRUCT(weights_host, dxy_bottom_right, i, j, (i == DIMY - 1 || j == DIMX - 1) ? 0.0 : dxy_1);

            SET_STRUCT(weights_host, dyx_top_left, i, j, (i == 0 || j == 0) ? 0.0 : dyx_1);
            SET_STRUCT(weights_host, dyx_top_right, i, j, (i == 0 || j == DIMX - 1) ? 0.0 : dyx_1);
            SET_STRUCT(weights_host, dyx_bottom_left, i, j, (i == DIMY - 1 || j == 0) ? 0.0 : dyx_1);
            SET_STRUCT(weights_host, dyx_bottom_right, i, j, (i == DIMY - 1 || j == DIMX - 1) ? 0.0 : dyx_1);            
    	}

       /* for (j = EDGE - EDGE_SIZE; j < EDGE + EDGE_SIZE; j++){
            SET_STRUCT(weights_host, next_i, i, j, i == DIMY - 1 ? 0.0 : 1.0 + (ratio - 1.0) * (j+1-EDGE+EDGE_SIZE)/2/EDGE_SIZE);
            SET_STRUCT(weights_host, next_j, i, j, j == DIMX - 1 ? 0.0 : ratio - (ratio - 1.0) * (j+1-EDGE+EDGE_SIZE)/2/EDGE_SIZE);
            SET_STRUCT(weights_host, prev_i, i, j, i == 0 ? 0.0 : 1.0 + (ratio - 1.0) * (j-EDGE+EDGE_SIZE)/2/EDGE_SIZE);
            SET_STRUCT(weights_host, prev_j, i, j, j == 0 ? 0.0 : ratio - (ratio - 1.0) * (j-EDGE+EDGE_SIZE)/2/EDGE_SIZE);
        }*/
            j = EDGE;
            SET_STRUCT(weights_host, next_i, i, j, i == DIMY - 1 ? 0.0 : dyy_1);
            SET_STRUCT(weights_host, next_j, i, j, j == DIMX - 1 ? 0.0 : dxx_2);
            SET_STRUCT(weights_host, prev_i, i, j, i == 0 ? 0.0 : dyy_1);
            SET_STRUCT(weights_host, prev_j, i, j, j == 0 ? 0.0 : dxx_1);

            SET_STRUCT(weights_host, dxy_top_left, i, j, (i == 0 || j == 0) ? 0.0 : dxy_1);
            SET_STRUCT(weights_host, dxy_top_right, i, j, (i == 0 || j == DIMX - 1) ? 0.0 : dxy_2);
            SET_STRUCT(weights_host, dxy_bottom_left, i, j, (i == DIMY - 1 || j == 0) ? 0.0 : dxy_1);
            SET_STRUCT(weights_host, dxy_bottom_right, i, j, (i == DIMY - 1 || j == DIMX - 1) ? 0.0 : dxy_2);

            SET_STRUCT(weights_host, dyx_top_left, i, j, (i == 0 || j == 0) ? 0.0 : dyx_1);
            SET_STRUCT(weights_host, dyx_top_right, i, j, (i == 0 || j == DIMX - 1) ? 0.0 : dyx_2);
            SET_STRUCT(weights_host, dyx_bottom_left, i, j, (i == DIMY - 1 || j == 0) ? 0.0 : dyx_1);
            SET_STRUCT(weights_host, dyx_bottom_right, i, j, (i == DIMY - 1 || j == DIMX - 1) ? 0.0 : dyx_2); 


            /*Don't forget to remove +1 below*/

    	for (j = EDGE + 1; j < DIMX; j++) {
    	    SET_STRUCT(weights_host, next_i, i, j, i == DIMY - 1 ? 0.0 : dyy_2);
    	    SET_STRUCT(weights_host, next_j, i, j, j == DIMX - 1 ? 0.0 : dxx_2);
    	    SET_STRUCT(weights_host, prev_i, i, j, i == 0 ? 0.0 : dyy_2);
    	    SET_STRUCT(weights_host, prev_j, i, j, j == 0 ? 0.0 : dxx_2);

            SET_STRUCT(weights_host, dxy_top_left, i, j, (i == 0 || j == 0) ? 0.0 : dxy_2);
            SET_STRUCT(weights_host, dxy_top_right, i, j, (i == 0 || j == DIMX - 1) ? 0.0 : dxy_2);
            SET_STRUCT(weights_host, dxy_bottom_left, i, j, (i == DIMY - 1 || j == 0) ? 0.0 : dxy_2);
            SET_STRUCT(weights_host, dxy_bottom_right, i, j, (i == DIMY - 1 || j == DIMX - 1) ? 0.0 : dxy_2);

            SET_STRUCT(weights_host, dyx_top_left, i, j, (i == 0 || j == 0) ? 0.0 : dyx_2);
            SET_STRUCT(weights_host, dyx_top_right, i, j, (i == 0 || j == DIMX - 1) ? 0.0 : dyx_2);
            SET_STRUCT(weights_host, dyx_bottom_left, i, j, (i == DIMY - 1 || j == 0) ? 0.0 : dyx_2);
            SET_STRUCT(weights_host, dyx_bottom_right, i, j, (i == DIMY - 1 || j == DIMX - 1) ? 0.0 : dyx_2); 
    	}
    }

    copy_struct(weights_dev, weights_host, HOST_TO_DEVICE);
    destroy_struct(weights_host, HOST);
}


void set_block_weights(weights_array_t *weights_dev)
{
    int i, j;
    weights_array_t *weights_host;

    int i1 = (int) (DIMX - XLENGTH)/2;
    int i2 = (int) (DIMX + XLENGTH)/2;
    int j1 = (int) (DIMY - YLENGTH)/2;
    int j2 = (int) (DIMY + YLENGTH)/2;

    weights_host = create_struct(HOST, weights_array_t);
    copy_struct(weights_host, weights_dev, DEVICE_TO_HOST);

 /* set inner weights - make sure they are equal for connections */
    for (j = j1; j < j2 - 1; j++)
	for( i = i1; i < i2; i++){
	    SET_STRUCT(weights_host, next_j, i, j, 0);
	    SET_STRUCT(weights_host, prev_j, i, j + 1, 0);
	}

    for (j = j1; j < j2; j++)
	for( i = i1; i < i2 -1; i++){
	    SET_STRUCT(weights_host, next_i, i, j, 0);
	    SET_STRUCT(weights_host, prev_i, i + 1, j , 0);
	}

    copy_struct(weights_dev, weights_host, HOST_TO_DEVICE);
    destroy_struct(weights_host, HOST);
}
