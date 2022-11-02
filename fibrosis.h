#ifndef FIBROSIS_H
#define FIBROSIS_H

#include "medium.h"

void set_fibrosis(char *medium, float prob, long seed);
void set_square_fibrosis(medium_t medium,
			 float mean, float hetero, int size, long seed);
void set_square_fibrosis2(medium_t medium, float mean, float hetero,
			  int size, int discr, long seed);
void set_ablation(medium_t medium, int x, int y, int r);

#endif	/* FIBROSIS */
