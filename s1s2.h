#ifndef S1S2_H
#define S1S2_H

#include "statevar.h"

void apply_stim(voltage_t u, int x0, int x1, int y0, int y1);
void set_s1s2_threshold(real t);
void set_s1s2_measure(int y, int x);
void s1s2(voltage_t u);
void apply_stim(voltage_t u, int x0, int x1, int y0, int y1);
void apply_circ_stim(voltage_t u, int x, int y, int r);
void apply_s1(voltage_t u);
void apply_s2(voltage_t u);

void apply_s1(voltage_t u);

#endif	/* S1S2_H */
