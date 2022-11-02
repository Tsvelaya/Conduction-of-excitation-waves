#ifndef STEP_H
#define STEP_H

#include "statevar.h"

#define BLOCK_DIMX 128
#define BLOCK_DIMY 1

#ifdef __cplusplus
extern "C"
#endif
void step(voltage_t volt, voltage_t volt_new, var_array_t *vars,
	  weights_array_t *weights);

#endif	/* STEP_H */
