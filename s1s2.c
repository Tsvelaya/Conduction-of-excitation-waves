#include "value.h"
#include "statevar.h"
#include "cuda.h"
#include "options.h"

#define STIM_STRENTH REAL(50.0)

typedef enum {
    FRESH,
    WAITING,
    PASSING,
    GONE,
    APPLIED
} point_state;

static real threshold = REAL(-70.0);
static int measure_x = DIMX / 2;
static int measure_y = DIMY / 2;
/*static point_state state = FRESH;*/

void apply_stim(voltage_t u, int x0, int x1, int y0, int y1)
{
    int i, j;
    voltage_t v;

    v = create_voltage(HOST);
    copy_voltage(v, u, DEVICE_TO_HOST);
    for (i = y0; i < y1; i++)
        for (j = x0; j < x1; j++)
            SET_VOLT(v, i, j, STIM_STRENTH);
    copy_voltage(u, v, HOST_TO_DEVICE);
    destroy_voltage(v, HOST);
}

void apply_circ_stim(voltage_t u, int x, int y, int r)
{
    int i, j;
    voltage_t v;

    v = create_voltage(HOST);
    copy_voltage(v, u, DEVICE_TO_HOST);
    for (i = 0; i < DIMY; i++)
	for (j = 0; j < DIMX; j++)
	    if (SQR(i - y) + SQR(j - x) <= SQR(r))
		SET_VOLT(v, i, j, STIM_STRENTH);
    copy_voltage(u, v, HOST_TO_DEVICE);
    destroy_voltage(v, HOST);
}


void apply_s1(voltage_t u)
{
    apply_stim(u, 0, 6, 0, DIMY);
}

void apply_s2(voltage_t u)
{
    apply_stim(u, 0, DIMX / 2, 0, DIMY / 2);
}


void set_s1s2_threshold(real t)
{
    threshold = t;
}

void set_s1s2_measure(int y, int x)
{
    measure_y = y;
    measure_x = x;
}

void s1s2(voltage_t u)
{
    static point_state s = FRESH;
    real volt;

    switch (s) {
    case FRESH:
        apply_s1(u);
        s = WAITING;
        return;
    case WAITING:
        xmemcpy(&volt, &GET_VOLT(u, measure_y, measure_x),
                sizeof(volt), DEVICE_TO_HOST);
        s = volt > threshold ? PASSING : WAITING;
        return;
    case PASSING:
        xmemcpy(&volt, &GET_VOLT(u, measure_y, measure_x),
                sizeof(volt), DEVICE_TO_HOST);
        s = volt < threshold ? GONE : PASSING;
        return;
    case GONE:
        apply_s2(u);
        s = APPLIED;
        return;
    case APPLIED:
        return;
    }
}

