#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "medium.h"

medium_t create_medium(location_t loc)
{
    return (medium_t) xmalloc(NUM_ELEMENTS * sizeof(real), loc);
}

void destroy_medium(medium_t medium, location_t loc)
{
    xfree(medium, loc);
}

void copy_medium(medium_t dst, medium_t src, direction_t dir)
{
    xmemcpy(dst, src, NUM_ELEMENTS * sizeof(real), dir);
}

void init_uniform_medium(char *medium)
{
    memset(medium, 1, NUM_ELEMENTS);
}

void write_medium(medium_t medium, char *filename)
{
    FILE *f;
    int dim[] = { DIMX, DIMY };
    size_t r;

    f = fopen(filename, "w");
    if (!f) { perror(filename); abort(); }

    r = fwrite(dim, sizeof(dim[0]), SIZE(dim), f);
    assert(r == SIZE(dim));
    r = fwrite(medium, 1, NUM_ELEMENTS, f);
    assert(r == NUM_ELEMENTS);

    fclose(f);
}

void read_medium(medium_t medium, char *filename)
{
    FILE *f;
    int dim[2];
    size_t r;

    f = fopen(filename, "r");
    if (!f) { perror(filename); abort(); }

    r = fread(dim, sizeof(dim[0]), SIZE(dim), f);
    assert(r == SIZE(dim));
    assert(dim[0] == DIMX && dim[1] == DIMY);

    r = fread(medium, 1, NUM_ELEMENTS, f);
    assert(r == NUM_ELEMENTS);
    fclose(f);
}

void set_weights_of_medium(weights_array_t *weights, medium_t medium)
{
    int i, j, k;
    real **m;

    m = (real **) weights;

    for (k = 0; k < sizeof(weights_array_t) / sizeof(real **); k++)
	memset(m[k], 0, NUM_ELEMENTS * sizeof(real));

    for (i = 0; i < DIMY; i++) {
	for (j = 0; j < DIMX; j++) {
	    if (!GET_ARRAY(medium, i, j)) {
		SET_STRUCT(weights, next_i, i, j, 0);
		SET_STRUCT(weights, next_j, i, j, 0);
		SET_STRUCT(weights, prev_i, i, j, 0);
		SET_STRUCT(weights, prev_j, i, j, 0);
	    } else {
		SET_STRUCT(weights, next_i, i, j,
			   i == DIMY - 1 ||
			     !GET_ARRAY(medium, i + 1, j) ? 0.0 : 1.0);
		SET_STRUCT(weights, next_j, i, j,
			   j == DIMX - 1 ||
			     !GET_ARRAY(medium, i, j + 1) ? 0.0 : 1.0);
		SET_STRUCT(weights, prev_i, i, j,
			   i == 0 ||
			     !GET_ARRAY(medium, i - 1, j) ? 0.0 : 1.0);
		SET_STRUCT(weights, prev_j, i, j,
			   j == 0 ||
			     !GET_ARRAY(medium, i, j - 1) ? 0.0 : 1.0);
	    }
	}
    }
}
