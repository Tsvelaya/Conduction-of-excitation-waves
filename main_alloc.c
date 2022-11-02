#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/time.h>

#include "debug.h"
#include "util.h"
#include "parameters.cuh"
#include "statevar.h"
#include "init.h"
#include "step.h"
#include "output.h"
#include "tabulate.cuh"
#include "statevar.h"

int verbose = 0;

int main()
{
    int i;
    var_array_t *v;
    var_array_t *dev_v, *dev_u;
    weights_array_t *weights, *dev_w;
    tab_options volt_tab = { TAB_VOLTAGE_MIN, TAB_VOLTAGE_MAX, TAB_POINTS };
    tab_options ca_tab = { TAB_CA_MIN, TAB_CA_MAX, TAB_POINTS };

    v = create_host_var_array();
    dev_v = create_device_var_array();
    dev_u = create_device_var_array();

    read_backup(v, "spiral.bin");
    /* copy_var_array_host_to_device(dev_v, v); */
    copy_var_array_host_to_device(dev_u, v);

    weights = create_host_weights_array();
    dev_w = create_device_weights_array();
    set_uniform_weights(weights);
    copy_weights_array_host_to_device(dev_w, weights);
    init_constants();
    init_tabulation(volt_tab, ca_tab);

    /* init_output_thread("results_test"); */

    for (i = 0; i < 5000; i++) {
	dprint(i);
	step(dev_u, dev_v, dev_w);
	copy_var_array_device_to_host(v, dev_u);
	step(dev_v, dev_u, dev_w);
	/* copy_var_array_device_to_host(v, dev_v); */
    }



    /* sync_output_thread(); */

    return 0;
}
