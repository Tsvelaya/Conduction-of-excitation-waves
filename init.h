#ifndef INIT_H
#define INIT_H

#include "statevar.h"

void init(voltage_t *u_dev, voltage_t *v_dev, var_array_t **vars_dev,
	  weights_array_t **weights_dev);
void reinit();

#endif /* DRIVER_H */
