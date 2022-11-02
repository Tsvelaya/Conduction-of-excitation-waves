#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/time.h>

#include "util.h"
#include "debug.h"
#include "parameters.cuh"
#include "statevar.h"
#include "init.h"
#include "step.h"
#include "output.h"
#include "tabulate.cuh"

int verbose = 0;
int device_num = 0;

int main(int argc, char *argv[])
{
    int opt;
    struct stat dst;
    char *options = "d:i:v";
    char *output_dir = "results";
    char *init_file = NULL;
    struct timeval tv;
    time_t time;

    int i, j;
    voltage_t u, v;
    var_array_t *vars;
    weights_array_t *wts;

    voltage_t volt_host;
    var_array_t *vars_host;

    while ((opt = getopt(argc, argv, options)) != -1) {
	switch (opt) {
	case 'd':
	    device_num = atoi(optarg);
	    break;
	case 'i':
	    init_file = optarg;
	    break;
        case 'v':
            verbose++;
            break;
	default:
	    exit(EXIT_FAILURE);
	}
    }

    if (optind != argc - 1) exit(EXIT_FAILURE);

    output_dir = argv[optind];
    switch (stat(output_dir, &dst)) {
    case 0:
	if (!S_ISDIR(dst.st_mode)) {
	    fprintf(stderr, "'%s' is not a directory\n", output_dir);
	    exit(EXIT_FAILURE);
	}
	break;
    default:
	if (mkdir(output_dir, 0777) != 0) {
	    perror(output_dir);
	    exit(EXIT_FAILURE);
	}
	break;
    }

    init(&u, &v, &vars, &wts);

    volt_host = create_voltage(HOST);
    vars_host = create_var_array(HOST);

    if (init_file != NULL) {
	read_backup(volt_host, vars_host, init_file);
	copy_voltage(u, volt_host, HOST_TO_DEVICE);
	copy_var_array(vars, vars_host, HOST_TO_DEVICE);

    } else {
	copy_voltage(volt_host, u, DEVICE_TO_HOST);
	for (i = 0; i < DIMY; i++)
	    for (j = 0; j < 5; j++)
		SET_VOLT(volt_host, i, j, 50.0);
	copy_voltage(u, volt_host, HOST_TO_DEVICE);
    }

    init_output_thread(output_dir);

    gettimeofday(&tv, NULL);
    time = tv.tv_sec;

    for (i = 0; i < 5000; i++) {
     	if (i % 400 == 0)
    	    output_picture(u);

     	step(u, v, vars, wts);
     	SWAP(u, v);
    }

    gettimeofday(&tv, NULL);
    printf("Took %lds\n", tv.tv_sec - time);

    sync_output_thread();

    copy_voltage(volt_host, u, DEVICE_TO_HOST);
    copy_var_array(vars_host, vars, DEVICE_TO_HOST);
    write_backup(volt_host, vars_host, "test_backup.bin");

    return EXIT_SUCCESS;
}
