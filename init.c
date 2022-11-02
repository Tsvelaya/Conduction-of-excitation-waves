#include "cuda.h"
#include "statevar.h"
#include "parameters.cuh"
#include "functions.h"
#include "tabulate.cuh"

void init(voltage_t *u_dev, voltage_t *v_dev, var_array_t **vars_dev,
	  weights_array_t **weights_dev)
{
    int i;
    tab_options volt_tab = { TAB_VOLTAGE_MIN, TAB_VOLTAGE_MAX, TAB_POINTS };
    tab_options ca_tab = { TAB_CA_MIN, TAB_CA_MAX, TAB_POINTS };
    voltage_t volt_host;
    var_array_t *vars_host;


    init_cuda();
    init_constants();
    init_tabulation(volt_tab, ca_tab);

    volt_host = create_voltage(HOST);
    for (i = 0; i < NUM_ELEMENTS; i++)
	volt_host[i] = VOLTAGE_INIT;

    *u_dev = create_voltage(DEVICE);
    *v_dev = create_voltage(DEVICE);

    copy_voltage(*u_dev, volt_host, HOST_TO_DEVICE);
    copy_voltage(*v_dev, volt_host, HOST_TO_DEVICE);

    destroy_voltage(volt_host, HOST);

    vars_host = create_struct(HOST, var_array_t);

#define INIT(member, value)				\
    for (i = 0; i < NUM_ELEMENTS; i++)			\
	(vars_host->member)[i] = (value)

#define INIT_GATE(member, gate_name)				\
    for (i = 0; i < NUM_ELEMENTS; i++)				\
	(vars_host->member)[i] = gate_name##_inf(VOLTAGE_INIT)

    INIT(Ca, CA_IN_INIT);
    INIT(Ca_JSR, REAL(778.1041)); INIT(Ca_NSR, REAL(799.8846));
    INIT(M, REAL(0.02272)); INIT(H, 0.24062); INIT(J, 0.20256);
    INIT(D, REAL(0.000302284)); INIT(F, REAL(0.99862)); INIT(FCa, REAL(0.99578));
    INIT(B, REAL(0.0027)); INIT(G, REAL(0.63674));
    INIT(Y, REAL(0.07277));
    INIT(R, REAL(0.00672)); INIT(S, REAL(0.97534)); INIT(Sslow, REAL(0.22548));
    INIT(Xr, REAL(0.025742210977));
    INIT(Xs1, REAL(0.012668791315)); INIT(Xs2, REAL(0.028399873909));
    INIT(P_o1, REAL(0.01315));

    *vars_dev = create_struct(DEVICE, var_array_t);
    copy_struct(*vars_dev, vars_host, HOST_TO_DEVICE);
    destroy_struct(vars_host, HOST);


    *weights_dev = create_struct(DEVICE, weights_array_t);
    set_uniform_weights(*weights_dev);

}

void reinit()
{
    init_constants();
    reinit_tabulation();
}
