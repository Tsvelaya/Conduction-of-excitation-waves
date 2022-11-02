#include "output.h"
#include "cuda.h"
#include "statevar.h"
#include "parameters.cuh"
#include "functions.h"

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <zlib.h>
#include <string.h>

#ifdef PNG
#include <png.h>
#endif

extern int verbose;

static int output_flag = 0;
static pthread_mutex_t output_flag_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t output_flag_cv = PTHREAD_COND_INITIALIZER;

static char *despath;
static real *voltage = NULL;
static real *volt_dev = NULL;

static inline unsigned char char_of_volt(real volt)
{
    return (unsigned char) (volt + 100.0);
}

#ifndef PNG
void write_picture(voltage_t volt, int k)
{
    int i, j;
    char file[BUFSIZ];
    gzFile f;

    snprintf(file, sizeof(file), "%s/PictureVoltageData%.4d.gz", despath, k);
    f = gzopen(file, "w");
    if (!f) { perror(file); abort(); }

    gzprintf(f, "%i\n", DIMX);
    gzprintf(f, "%i\n", DIMY);

    for (j = 0; j < DIMX; j++)
	for (i = 0; i < DIMY; i++)
	    gzprintf(f, "%c", char_of_volt(GET_VOLT(volt, i, j)));

    gzclose(f);
}

#else  /* ifdef PNG */

void **matrix(size_t n, size_t m, size_t s)
{
    void **a;
    int i;

    a = malloc(n * (sizeof(void*) + m * s));
    if (a)
        for (i=0; i<n; ++i)
            a[i] = (char *)(a + n) + i * m * s;
    return a;
}

void write_picture(voltage_t volt, int k)
{
    FILE *f;
    png_structp pp;
    png_infop ip;
    char file[BUFSIZ];
    int i, front, front2, front3, front4;
    double sum;
    static unsigned char **buf = NULL;

    if (buf == NULL)
	buf = (unsigned char **) matrix(DIMY, DIMX, sizeof(unsigned char));

    snprintf(file, sizeof(file), "%s/Movie%.4d.png", despath, k);
    f = fopen(file, "w");
    if (!f) { perror(file); abort(); }

    pp = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    ip = png_create_info_struct(pp);

    png_init_io(pp, f);
    png_set_IHDR(pp, ip, DIMX, DIMY, 8, PNG_COLOR_TYPE_GRAY,
		 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
		 PNG_FILTER_TYPE_DEFAULT);

    for (i = 0; i < NUM_ELEMENTS; i++)
        buf[0][i] = char_of_volt(volt[i]);

    png_set_rows(pp, ip, buf);
    png_write_png(pp, ip, PNG_TRANSFORM_IDENTITY, NULL);
    png_write_end(pp, ip);
    png_destroy_write_struct(&pp, &ip);

    fclose(f);


    /*FRONT detector*/

    /*snprintf(file, sizeof(file), "%s/log_%.3f_%.3f.txt", despath,ratio,frequency);*/
    snprintf(file, sizeof(file), "%s/log_front.txt", despath);
    if(k==0) f = fopen(file,"w");
    f = fopen(file,"a");
    front = 0;
    for (i = 0; i<DIMX-10; i++)
       if(volt[ LINE * DIMX + i ] > THR_FRONT && volt[ LINE * DIMX + i +1 ] <= THR_FRONT 
        && abs(i-EDGE) < abs(front - EDGE) )
        front=i; 
    fprintf(f,"%d\n",front);
    fclose(f);

    /*Check integral*/

    /*snprintf(file, sizeof(file), "%s/log.txt", despath);
    if(k==0) f = fopen(file,"w");
    f = fopen(file,"a");
    sum = 0;
    for (i = 0; i < NUM_ELEMENTS; i++)
            sum+=volt[i];
    fprintf(f,"%g\n",sum);
    fclose(f);*/

    /*AP*/

    snprintf(file, sizeof(file), "%s/log_AP.txt", despath);
    if(k==0) f = fopen(file,"w");
    f = fopen(file,"a");
    fprintf(f,"%g\n",volt[10]);
    fclose(f);
}

#endif

void write_actmap(actmap_t act)
{
    int i, j;
    FILE *f;
    char file[BUFSIZ];

    snprintf(file, sizeof(file), "%s/activation", despath);
    f = fopen(file, "w");
    if (!f) { perror(file); abort(); }

    for (i = 0; i < DIMY; i++) {
	for (j = 0; j < DIMX; j++)
	    fprintf(f, "%d ", GET_VOLT(act, i, j));
	fprintf(f, "\n");
    }

    fclose(f);
}


/* void write_point_backup_data(var_array_t *local, int i, int j, int step) */
/* { */
/*     static FILE *f = NULL; */
/*     double time = step * TIMESTEP; */

/*     if (f == NULL) { */
/* 	char filename[BUFSIZ]; */
/* 	snprintf(filename, sizeof(filename), */
/* 		 "%s/PointData.I%d.J%d", despath, i, j); */
/* 	f = fopen(filename, "w"); */
/* 	if (!f) { perror(filename); abort(); } */
/*     } */

/*     fprintf(f, "%.20g ", time);                    /\*1*\/ */
/*     fprintf(f, "%.20g ", GET_VAR(local, Voltage,i,j)); /\*2*\/ */
/*     fprintf(f, "%.20g ", */
/* 	    ca_i_free_of_total(GET_VAR(local, Cai_total,i,j)));     /\*3*\/ */
/*     fprintf(f, "%.20g ", */
/* 	    ca_sr_free_of_total(GET_VAR(local, CaSR_total,i,j)));    /\*4*\/ */
/*     fprintf(f, "%.20g ", */
/* 	    ca_ss_free_of_total(GET_VAR(local, CaSS_total,i,j)));    /\*5*\/ */
/*     fprintf(f, "%.20g ", GET_VAR(local, M,i,j));       /\*6*\/ */
/*     fprintf(f, "%.20g ", GET_VAR(local, H,i,j));       /\*7*\/ */
/*     fprintf(f, "%.20g ", GET_VAR(local, J,i,j));       /\*8*\/ */
/*     fprintf(f, "%.20g ", GET_VAR(local, Xr1,i,j));     /\*9*\/ */
/*     fprintf(f, "%.20g ", GET_VAR(local, Xr2,i,j));     /\*10*\/ */
/*     fprintf(f, "%.20g ", GET_VAR(local, Xs,i,j));      /\*11*\/ */
/*     fprintf(f, "%.20g ", GET_VAR(local, R,i,j));       /\*12*\/ */
/*     fprintf(f, "%.20g ", GET_VAR(local, S,i,j));       /\*13*\/ */
/*     fprintf(f, "%.20g ", GET_VAR(local, D,i,j));       /\*14*\/ */
/*     fprintf(f, "%.20g ", GET_VAR(local, F,i,j));       /\*15*\/ */
/*     fprintf(f, "%.20g ", GET_VAR(local, F2,i,j));      /\*16*\/ */
/*     fprintf(f, "%.20g ", GET_VAR(local, FCaSS,i,j));   /\*17*\/ */
/*     fprintf(f, "%.20g ", GET_VAR(local, Rbar,i,j));    /\*18*\/ */
/*     fprintf(f, "\n"); */
/*     fflush(f); */
/* } */

void *write_thread(void *pvolt)
{
    int counter = 0;
    voltage_t volt = (real *)pvolt;

    while (1) {
	pthread_mutex_lock(&output_flag_mutex);
	while (output_flag == 0)
	    pthread_cond_wait(&output_flag_cv, &output_flag_mutex);
	if (verbose) {
            printf("step = %d\n", counter);
            fflush(stdout);
        }
	copy_voltage(volt, volt_dev, DEVICE_TO_HOST_ASYNC);
	write_picture(volt, counter);
	output_flag = 0;
	pthread_cond_signal(&output_flag_cv);
	pthread_mutex_unlock(&output_flag_mutex);
	counter++;
    }

    return NULL;
}

void init_output_thread(char *dirname)
{
    pthread_t id;

    if (voltage != NULL) {
	fprintf(stderr, "Output thread has already been initialized\n");
	abort();
    }

    despath = dirname;

    volt_dev = create_voltage(DEVICE);
    voltage = create_voltage(LOCKED);

    pthread_create(&id, NULL, write_thread, (void *)voltage);
}

void output_picture(voltage_t src)
{
    pthread_mutex_lock(&output_flag_mutex);
    while (output_flag == 1)
	pthread_cond_wait(&output_flag_cv, &output_flag_mutex);
    copy_voltage(volt_dev, src, DEVICE_TO_DEVICE);
    output_flag = 1;
    pthread_cond_signal(&output_flag_cv);
    pthread_mutex_unlock(&output_flag_mutex);
}

void sync_output_thread()
{
    pthread_mutex_lock(&output_flag_mutex);
    while (output_flag == 1)
        pthread_cond_wait(&output_flag_cv, &output_flag_mutex);
    pthread_mutex_unlock(&output_flag_mutex);
}

void write_backup(voltage_t volt, var_array_t *var, char *filename)
{
    int i, k;
    gzFile f;
    real **m;
    float tmp;

    m = (real **) var;

    f = gzopen(filename, "w");
    if (!f) { perror(filename); abort(); }

    for (i = 0; i < NUM_ELEMENTS; i++) {
	tmp = (float) volt[i];
	gzwrite(f, &tmp, sizeof(tmp));
    }

    for (k = 0; k < sizeof(var_array_t) / sizeof(real **); k++) {
	for (i = 0; i < NUM_ELEMENTS; i++) {
	    tmp = (float) m[k][i];
	    gzwrite(f, &tmp, sizeof(tmp));
	}
    }

    gzclose(f);
}

void read_backup(voltage_t volt, var_array_t *var, char *filename)
{
    int i, k;
    gzFile f;
    real **m;
    float tmp;

    m = (real **)var;

    f = gzopen(filename, "r");
    if (!f) { perror(filename); abort(); }

    for (i = 0; i < NUM_ELEMENTS; i++) {
	gzread(f, &tmp, sizeof(tmp));
	volt[i] = (real) tmp;
    }

    for (k = 0; k < sizeof(var_array_t) / sizeof(real **); k++) {
	for (i = 0; i < NUM_ELEMENTS; i++) {
	    gzread(f, &tmp, sizeof(tmp));
	    m[k][i] = (real) tmp;

	}
    }

    gzclose(f);
}
