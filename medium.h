#ifndef MEDIUM_H
#define MEDIUM_H

#include "statevar.h"

typedef char *medium_t;

medium_t create_medium(location_t loc);
void destroy_medium(medium_t m, location_t loc);
void copy_medium(medium_t dst, medium_t src, direction_t dir);

void write_medium(medium_t m, char *filename);
void read_medium(medium_t m, char *filename);

void init_uniform_medium(medium_t m);
void set_weights_of_medium(weights_array_t *weights, medium_t medium);

#endif	/* MEDIUM_H */
