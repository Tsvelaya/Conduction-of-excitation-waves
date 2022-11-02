#ifndef OUTPUT_H
#define OUTPUT_H

#include "statevar.h"
#include "activation.h"
#include "options.h"

void init_output_thread(char *dirname);
void output_picture(voltage_t volt);
void sync_output_thread();
void write_point_backup_data(var_array_t *local, int i, int j, int step);
void write_backup(voltage_t volt, var_array_t *var, char *filename);
void read_backup(voltage_t volt, var_array_t *var, char *filename);

void write_actmap(actmap_t act);

#endif	/* OUTPUT_H */
