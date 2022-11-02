#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "statevar.h"
#include "util.h"
#include "medium.h"

typedef int *actmap_t;

#ifdef __cplusplus
extern "C" {
#endif
void activation(voltage_t volt, medium_t med, int step);
actmap_t get_activation();

#ifdef __cplusplus
}
#endif

#endif	/* ACTIVATION_H */
