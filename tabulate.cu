//                                                                   -*- C++ -*-

#include "tabulate.cuh"
#include "functions.h"

#include "debug.h"

__constant__ tabulation_t Rev_potential_ca;

__constant__ tabulation_t M_inf, M_tau_er, H_inf, H_tau_er, J_inf, J_tau_er,
    Xr_inf, Xr_tau_er, Xs1_inf, Xs1_tau_er, Xs2_inf, Xs2_tau_er,
    R_inf, R_tau_er, S_inf, S_tau_er, Sslow_inf, Sslow_tau_er,
    D_inf, D_tau_er, F_inf, F_tau_er, FCa_inf, B_inf, 
    B_tau_er, G_inf, G_tau_er, Y_inf, Y_tau_er, G_cal_first_abs,
    G_cal_second_abs, G_cal_first, G_cal_second;

__constant__ tabulation_t Kna_pump;

#define CONNECT(A, a, suffix) do {					\
	tabulation_t t = tabulation(a##_##suffix, min, max, n);		\
	cudaError_t err = cudaMemcpyToSymbol(A##_##suffix, &t, sizeof(t)); \
	check(err);							\
    } while (0)
#define RECONNECT(A, a, suffix) do {                                    \
        tabulation_t t;                                                 \
        cudaError_t e;                                                  \
        e = cudaMemcpyFromSymbol(&t, A##_##suffix, sizeof(t));          \
        check(e);                                                       \
        retabulate(&t, a##_##suffix, t.min, t.max, t.size);             \
        e = cudaMemcpyToSymbol(A##_##suffix, &t, sizeof(t));            \
        check(e);                                                       \
    } while (0)

void tabulate_volt(real min, real max, int n)
{
    CONNECT(M, m, inf);
    CONNECT(M, m, tau_er);
    CONNECT(H, h, inf);
    CONNECT(H, h, tau_er);
    CONNECT(J, j, inf);
    CONNECT(J, j, tau_er);
    CONNECT(Xr, xr, inf);
    CONNECT(Xr, xr, tau_er);
    CONNECT(Xs1, xs1, inf);
    CONNECT(Xs1, xs1, tau_er);
    CONNECT(Xs2, xs2, inf);
    CONNECT(Xs2, xs2, tau_er);
    CONNECT(R, r, inf);
    CONNECT(R, r, tau_er);
    CONNECT(S, s, inf);
    CONNECT(S, s, tau_er);
    CONNECT(Sslow, sslow, inf);
    CONNECT(Sslow, sslow, tau_er);
    CONNECT(D, d, inf);
    CONNECT(D, d, tau_er);
    CONNECT(F, f, inf);
    CONNECT(F, f, tau_er);
    CONNECT(FCa, fca, inf);
    CONNECT(G, g, inf);
    CONNECT(G, g, tau_er);
    CONNECT(B, b, inf);
    CONNECT(B, b, tau_er);
    CONNECT(Y, y, inf);
    CONNECT(Y, y, tau_er);
    CONNECT(G, g, cal_first);
    CONNECT(G, g, cal_second);
    CONNECT(G, g, cal_first_abs);
    CONNECT(G, g, cal_second_abs);
}

void tabulate_ca(real min, real max, int n)
{
    CONNECT(Rev, rev, potential_ca);
}

#undef CONNECT

void init_tabulation(tab_options volt, tab_options ca)
{
    tabulate_volt(volt.min, volt.max, volt.size);
    tabulate_ca(ca.min, ca.max, ca.size);
}

void reinit_tabulation()
{
    RECONNECT(M, m, inf);
    RECONNECT(M, m, tau_er);
    RECONNECT(H, h, inf);
    RECONNECT(H, h, tau_er);
    RECONNECT(J, j, inf);
    RECONNECT(J, j, tau_er);
    RECONNECT(Xr, xr, inf);
    RECONNECT(Xr, xr, tau_er);
    RECONNECT(Xs1, xs1, inf);
    RECONNECT(Xs1, xs1, tau_er);
    RECONNECT(Xs2, xs2, inf);
    RECONNECT(Xs2, xs2, tau_er);
    RECONNECT(R, r, inf);
    RECONNECT(R, r, tau_er);
    RECONNECT(S, s, inf);
    RECONNECT(S, s, tau_er);
    RECONNECT(Sslow, sslow, inf);
    RECONNECT(Sslow, sslow, tau_er);
    RECONNECT(D, d, inf);
    RECONNECT(D, d, tau_er);
    RECONNECT(F, f, inf);
    RECONNECT(F, f, tau_er);
    RECONNECT(FCa, fca, inf);
    RECONNECT(G, g, inf);
    RECONNECT(G, g, tau_er);
    RECONNECT(B, b, inf);
    RECONNECT(B, b, tau_er);
    RECONNECT(Y, y, inf);
    RECONNECT(Y, y, tau_er);
    RECONNECT(G, g, cal_first);
    RECONNECT(G, g, cal_second);
    RECONNECT(G, g, cal_first_abs);
    RECONNECT(G, g, cal_second_abs);

    RECONNECT(Rev, rev, potential_ca);
}

#undef RECONNECT
