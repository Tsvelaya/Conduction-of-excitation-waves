/*                                                                -*- C -*-  -*/
#ifndef TABULATE_CUH
#define TABULATE_CUH

#ifdef __CUDACC__

#include "tabulation.cuh"

extern __constant__ tabulation_t Rev_potential_ca;

extern __constant__ tabulation_t M_inf, M_tau_er, H_inf, H_tau_er, J_inf, J_tau_er,
    Xr_inf, Xr_tau_er, Xs1_inf, Xs1_tau_er, Xs2_inf, Xs2_tau_er,
    R_inf, R_tau_er, S_inf, S_tau_er, Sslow_inf, Sslow_tau_er,
    D_inf, D_tau_er, F_inf, F_tau_er, FCa_inf, B_inf, B_tau_er,
    G_inf, G_tau_er, Y_inf, Y_tau_er, G_cal_first_abs,
    G_cal_second_abs, G_cal_first, G_cal_second;

extern __constant__ tabulation_t Kna_pump;

#endif

typedef struct {
    real min, max;
    int size;
} tab_options;

#ifdef __cplusplus
extern "C" {
#endif
void init_tabulation(tab_options volt, tab_options ca);
void reinit_tabulation();
#ifdef __cplusplus
}
#endif

#endif
