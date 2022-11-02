#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "activation.h"
#include "fibrosis.h"
#include "init.h"
#include "options.h"
#include "output.h"
#include "parameters.cuh"
#include "s1s2.h"
#include "statevar.h"
#include "step.h"
#include "tabulate.cuh"
#include "util.h"
#include "functions.h"


int main(int argc, char *argv[])
{

    struct timeval tv;
    time_t time;
    char backupfile[BUFSIZ];
    char backupfile1[BUFSIZ];
    char backupfile2[BUFSIZ];
    char backupfile3[BUFSIZ];

    int i, ic, j, j_max, address;
    real current, current_max, voltage_max;

    voltage_t u_dev, v_dev;
    voltage_t v_host;

    var_array_t *vars_dev;
    var_array_t *vars_host;
    weights_array_t *wts_dev;
    weights_array_t *wts_host;
    medium_t med_host;

    parse_options(argc, argv);
    ic = (int) (1000/frequency/TIMESTEP);

    init(&u_dev, &v_dev, &vars_dev, &wts_dev);
    v_host = create_voltage(HOST);
    vars_host = create_struct(HOST, var_array_t);

    wts_host = create_struct(HOST, weights_array_t);
    med_host = create_medium(HOST);
    init_uniform_medium(med_host);

    if (medium_file != NULL)
        read_medium(med_host, medium_file);

    if (fibrosis.size > 0)
	set_square_fibrosis2(med_host, fibrosis.mean, fibrosis.hetero,
			     fibrosis.size, fibrosis.discr, fibrosis.seed);

    if (ablation.r > 0)
        set_ablation(med_host, ablation.x, ablation.y, ablation.r);

    if (init_file != NULL) {
	read_backup(v_host, vars_host, init_file);
	copy_voltage(u_dev, v_host, HOST_TO_DEVICE);
	copy_struct(vars_dev, vars_host, HOST_TO_DEVICE);
    } else {
        for (i = 0; i < DIMY; i++)
            if (med_host[ADDRESS(i, DIMX / 2)]) {
                set_s1s2_measure(i, DIMX / 2);
                break;
            }
    }

    set_weights_of_medium(wts_host, med_host); 
    /* copy_struct(wts_dev, wts_host, HOST_TO_DEVICE); */
    set_half_anisotropy(wts_dev, ratio);

    if(wait == 0)
    switch (proto.type) {
    case UNDEF:
	if (init_file == NULL)
	    proto.type = S1S2;
	else
	    proto.type = CLEAR;
	break;
    case S1:
	apply_s1(u_dev);
	break;
    case RIGHT:
	apply_stim(u_dev, DIMX - 6, DIMX, 0, DIMY);
	break;
    case TOP:
    apply_stim(u_dev, 0, DIMX, 0, 6);
    break;
    case CIRC:
        apply_circ_stim(u_dev, proto.data.circ.x, proto.data.circ.y,
			proto.data.circ.r);
	break;
    case RECT:
	apply_stim(u_dev, proto.data.rect.x0, proto.data.rect.x1,
		   proto.data.rect.y0, proto.data.rect.y1);
	break;
    case CLEAR: case S1S2: default:
	break;
    }

    if (pretend) return 0;

    init_output_thread(output_dir);

    gettimeofday(&tv, NULL);
    time = tv.tv_sec;

    /*printf("E_Na: %.5f\n",E_Na);*/

    printf("AR = %.3f, frq = %.3f;\n",ratio,frequency);

    for (i = 0; i < stop; i++) {
        if (proto.type == S1S2)
            s1s2(u_dev);


	/*if (i == 340000){
             copy_voltage(v_host, u_dev, DEVICE_TO_HOST);
             copy_struct(vars_host, vars_dev , DEVICE_TO_HOST);
	     snprintf(backupfile1, sizeof(backupfile1), "%s/%s", output_dir, "backup1.bin");
    	     write_backup(v_host, vars_host, backupfile1);}

	if (i == 380000){
             copy_voltage(v_host, u_dev, DEVICE_TO_HOST);
             copy_struct(vars_host, vars_dev , DEVICE_TO_HOST);
	     snprintf(backupfile2, sizeof(backupfile2), "%s/%s", output_dir, "backup2.bin");
    	     write_backup(v_host, vars_host, backupfile2);}

	if (i == 480000){
             copy_voltage(v_host, u_dev, DEVICE_TO_HOST);
             copy_struct(vars_host, vars_dev , DEVICE_TO_HOST);
	     snprintf(backupfile3, sizeof(backupfile3), "%s/%s", output_dir, "backup3.bin");
    	     write_backup(v_host, vars_host, backupfile3);}*/



     	if (i % 5000 == 0)
     	     output_picture(u_dev);

        if (i % 500 == 0){
             copy_struct(vars_host, vars_dev , DEVICE_TO_HOST);
             copy_voltage(v_host, u_dev, DEVICE_TO_HOST);

           // current_max=0; j_max=0; voltage_max=-80;

           // for(j = 0; j < DIMX; j++){
                address = ADDRESS(0, 64);
                current = G_NA * CUBE(GET_STRUCT_ADDRESS(vars_host, M, address)) *
                GET_STRUCT_ADDRESS(vars_host, H, address) *
                GET_STRUCT_ADDRESS(vars_host, J, address) * (GET_VOLT_ADDRESS(v_host, address) - ENa);
                printf("%.4f, ", current);
}
                //current = (REAL(0.103) * (K_OUT/(K_OUT + REAL(210.0))) * (GET_VOLT_ADDRESS(v_host, address) - EK - REAL(6.1373)) / (REAL(0.1653) 
    //+ exp(REAL(0.0319) * (GET_VOLT_ADDRESS(v_host, address) - EK - REAL(6.1373)))));

                /*current = G_K1*xk1_abs(GET_VOLT_ADDRESS(v_host, address)) * (GET_VOLT_ADDRESS(v_host, address) - EK);
                if (current>current_max){
                    current_max = current;
                    j_max = j;
                }
            }
                address = ADDRESS(0, 0);
                voltage_max = GET_VOLT_ADDRESS(v_host, address);
            printf("M_%d: %.4f, %d, V: %f\n",i/1000, current_max,j_max,voltage_max);
        }*/

        if(proto.type == CIRC && i <= number_of_waves*ic && i % ic == 0 && i > wait)
            apply_circ_stim(u_dev, proto.data.circ.x, proto.data.circ.y,
            proto.data.circ.r);

        if(proto.type == S1 /*&& i <= number_of_waves*ic*/ && i % ic == 0 && i > wait)
            apply_s1(u_dev);

	if (meas_activation)
	    activation(u_dev, med_host, i);

     	step(u_dev, v_dev, vars_dev, wts_dev);
     	SWAP(u_dev, v_dev);
    }

    gettimeofday(&tv, NULL);
    printf("Took %lds\n", tv.tv_sec - time);

    sync_output_thread();

    if (meas_activation) {
	write_actmap(get_activation());
    }

    copy_voltage(v_host, u_dev, DEVICE_TO_HOST);
    copy_struct(vars_host, vars_dev , DEVICE_TO_HOST);

    /*int address = ADDRESS(15, 15);
    printf("\nM: %.4f\n",G_NA * CUBE(GET_STRUCT_ADDRESS(vars_host, M, address)) *
    GET_STRUCT_ADDRESS(vars_host, H, address) *
    GET_STRUCT_ADDRESS(vars_host, J, address) * (GET_VOLT_ADDRESS(v_host, address) - E_Na));*/

    snprintf(backupfile, sizeof(backupfile), "%s/%s", output_dir, "backup.bin");
    write_backup(v_host, vars_host, backupfile);
    snprintf(backupfile, sizeof(backupfile), "%s/%s", output_dir, "medium.bin");
    write_medium(med_host, backupfile);

    return EXIT_SUCCESS;
}
