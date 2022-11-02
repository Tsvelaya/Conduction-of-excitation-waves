/*                                                                  -*- C -*- */
#ifndef FUNCTIONS_TCC
#define FUNCTIONS_TCC

#include "value.h"
#include "parameters.cuh"
#include "debug.h"

#ifdef __CUDACC__
typedef real value;
#define CONST(a) (const_##a)
#define PREFIX __device__
#define INLINE
#else
typedef double value;
#undef REAL
#define REAL(a) (a)
#undef sqrt
#undef exp
#undef log
#undef copysign
#define CONST
#define PREFIX
#define INLINE static inline
#endif

PREFIX INLINE value choice(value left, value right, value threshold, value x)
{
    return 0.5 * (right + left + (right - left) * copysign(1.0, x - threshold));
}

PREFIX value m_inf(value v) //m_infinity 
{
    return REAL(1.0) / (REAL(1.0) + exp((REAL(45.0) + v) / (-REAL(6.5)))); 
}

PREFIX INLINE value tau_m(value v) //t_m
{
    return REAL(1.36) / (REAL(0.32) * (v + REAL(47.13)) / (REAL(1.0) - exp(-REAL(0.1) * (v + REAL(47.13)))) 
    + REAL(0.08) * exp(-v / REAL(11.0)));
}

PREFIX value m_tau_er(value v)
{
    return exp(- TIMESTEP / tau_m(v));
}

PREFIX value h_inf(value v) // h_infinity
{
    return REAL(1.0) / (REAL(1.0) + exp((REAL(76.1) + v) / REAL(6.07)));

}

PREFIX INLINE value tau_h(value v) // t_h
{
    return v >= -REAL(40.0) ? REAL(0.4537) * (REAL(1.0) + exp((v + REAL(10.66)) / (-REAL(11.1)))) : 
    REAL(3.49) / (REAL(0.135) * exp((v + REAL(80.0)) / (-REAL(6.8))) + REAL(3.56) * exp(REAL(0.079) * v) 
    + REAL(310000.0) * exp(REAL(0.35) * v));
}

PREFIX value h_tau_er(value v)
{
    return exp(- TIMESTEP / tau_h(v));
}

PREFIX value j_inf(value v) // j_infinity
{
    return h_inf(v);
}

PREFIX INLINE value tau_j1(value v) //t_j1 if V more -40
{
    return REAL(11.63) * (REAL(1.0) + exp(-REAL(0.1) * (v + REAL(32.0)))) / exp(-REAL(2.535) * pow(REAL(10.0),-REAL(7.0)) * v);
}

PREFIX INLINE value tau_j2(value v) //t_j2 if V less -40
{
    return REAL(3.49) / (((v + REAL(37.78)) / (REAL(1.0) + exp(REAL(0.311) * (v + REAL(79.23))))) * (-REAL(127140.0) 
    * exp(REAL(0.2444) * v) - REAL(3.474) * pow(REAL(10.0),-REAL(5.0)) * exp(-REAL(0.04391) * v)) + REAL(0.1212) 
    * exp(-REAL(0.01052) * v) / (REAL(1.0) + exp(-REAL(0.1378) * (v + REAL(40.14)))));
}

PREFIX INLINE value tau_j(value v) // t_j
{
    return v >= -REAL(40.0) ? tau_j1(v) : tau_j2(v);
}

PREFIX value j_tau_er(value v)
{
    return exp(- TIMESTEP / tau_j(v));
}

PREFIX value y_inf(value v)
{
    return REAL(1.0) / (REAL(1.0) + exp((v + REAL(78.65)) / REAL(6.33)));
}

PREFIX INLINE value tau_y(value v)
{
    return REAL(1000.0) / (REAL(0.11885) * exp((v + REAL(75.0)) / REAL(28.37)) + REAL(0.56236) * exp((v + REAL(75.0)) / (-REAL(14.19))));
}

PREFIX value y_tau_er(value v)
{
    return exp(- TIMESTEP / tau_y(v));
}

PREFIX value r_inf(value v)
{
    return REAL(1.0) / (REAL(1.0) + exp(-(v - REAL(3.55716)) / REAL(14.61299)));
}

PREFIX INLINE value tau_r(value v)
{
    return REAL(1000.0) /(REAL(45.16) * exp(REAL(0.03577) * (v + REAL(50.0))) + REAL(98.9) * exp(-REAL(0.1) * (v + REAL(38.0))));
}

PREFIX value r_tau_er(value v)
{
    return exp(- TIMESTEP / tau_r(v));
}

PREFIX value s_inf(value v)
{
    return REAL(1.0) / (REAL(1.0) + exp((v + REAL(31.97156)) / REAL(4.64291)));
}

PREFIX INLINE value tau_s(value v)
{
    return REAL(1000.0) * (REAL(0.35) * exp(-REAL(1.0) * pow(((v + REAL(70.0)) / REAL(15.0)),REAL(2.0))) + REAL(0.035)) - REAL(26.9);
}

PREFIX value s_tau_er(value v)
{
    return exp(- TIMESTEP / tau_s(v));
}

PREFIX value sslow_inf(value v)
{
    return s_inf(v);
}

PREFIX INLINE value tau_sslow(value v)
{
    return REAL(1000.0) * (REAL(3.7) * exp(-REAL(1.0) * pow(((v + REAL(70.0))/REAL(30.0)), REAL(2.0))) + REAL(0.035)) + REAL(37.4);
}

PREFIX value sslow_tau_er(value v)
{
    return exp(- TIMESTEP / tau_sslow(v));
}

PREFIX value xs1_inf(value v)
{
    return REAL(1.0) / (REAL(1.0) + exp(-(v - REAL(1.5)) / REAL(16.7)));
}

PREFIX INLINE value c1(value v)
{
    return REAL(7.19) * pow(REAL(10.0),-REAL(5.0)) * (v + REAL(30.0)) / (REAL(1.0) - exp(-REAL(0.148) * (v + REAL(30.0))));
}

PREFIX INLINE value c2(value v)
{
    return REAL(1.31) * pow(REAL(10.0),-REAL(4.0)) * (v + REAL(30.0)) / (exp(REAL(0.0687) * (v + REAL(30.0))) - REAL(1.0));
}

PREFIX INLINE value tau_xs1(value v)
{
    return REAL(1.0) / (c1(v) + c2(v));
}

PREFIX value xs1_tau_er(value v)
{
    return exp(- TIMESTEP / tau_xs1(v));
}

PREFIX value xs2_inf(value v)
{
    return xs1_inf(v);
}

PREFIX INLINE value tau_xs2(value v)
{
    return REAL(4.0) * tau_xs1(v);
}

PREFIX value xs2_tau_er(value v)
{
    return exp(- TIMESTEP / tau_xs2(v));
}

PREFIX value xr_inf(value v)
{
    return REAL(1.0) / (REAL(1.0) + exp(-(v + REAL(21.5)) / REAL(7.5)));
}

PREFIX INLINE value xk1_alpha(value v)
{
    return REAL(0.1) / (REAL(1.0) + exp(REAL(0.06) *
                (v - CONST(EK) - REAL(200.0))));
}

PREFIX INLINE value xk1_beta(value v)
{
    return (REAL(3.0) * exp(REAL(0.0002) * (v - CONST(EK) + REAL(100.0))) +
            exp(REAL(0.1) * (v - CONST(EK) - REAL(10.0)))) /
         (REAL(1.0) + exp(- REAL(0.5) * (v - CONST(EK))));
}
PREFIX value xk1_abs(value v)
{
    return xk1_alpha(v) / (xk1_alpha(v) + xk1_beta(v));
}

PREFIX INLINE value tau_xr(value v)
{
    return REAL(1.0) / ((REAL(0.00138) * (v + REAL(14.2)) / (REAL(1.0) - exp(-REAL(0.123) * (v + REAL(14.2)))))
    + (REAL(0.00061) * (v + REAL(38.9)) / (exp(REAL(0.145) * (v + REAL(38.9))) - REAL(1.0))));
}

PREFIX value xr_tau_er(value v)
{
    return exp(- TIMESTEP / tau_xr(v));
}

PREFIX value d_inf(value v)
{
    return REAL(1.0) / (REAL(1.0) + exp((-REAL(11.1) - v) / REAL(7.2)));
}

PREFIX INLINE value d_alpha(value v)
{
    return REAL(0.25) + REAL(1.4) / (REAL(1.0) + exp((-REAL(35.0) - v) / REAL(13.0)));
}

PREFIX INLINE value d_beta(value v)
{
    return REAL(1.4) / (REAL(1.0) + exp((v + REAL(5.0)) / REAL(5.0)));
}

PREFIX INLINE value d_gamma(value v)
{
    return REAL(1.0) / (REAL(1.0) + exp((REAL(50.0) - v) / REAL(20.0)));
}

PREFIX value d_tau_er(value v)
{
    return exp(- TIMESTEP / (d_alpha(v) * d_beta(v) + d_gamma(v)));
}

PREFIX value f_inf(value v)
{
    return REAL(1.0) / (REAL(1.0) + exp((REAL(23.3) + v) / REAL(5.4)));
}

PREFIX INLINE value tau_f(value v)
{
    return REAL(1125.0) * exp(-REAL(1.0) * pow((v + REAL(27.0)),REAL(2.0)) / REAL(240.0)) +
        REAL(165.0) / (REAL(1.0) + exp((REAL(25.0) - v) / REAL(10.0))) + REAL(120.0);
}

PREFIX value f_tau_er(value v)
{
    return exp(- TIMESTEP /  tau_f(v));
}

PREFIX INLINE value fca_alpha(value ca_in)
{
    return REAL(1.0) / (REAL(1.0) + pow(ca_in / REAL(0.325), REAL(8.0)));
}

PREFIX INLINE value fca_beta(value ca_in)
{
    return REAL(0.1) / (REAL(1.0) + exp((ca_in - REAL(0.5)) / REAL(0.1)));
}

PREFIX INLINE value fca_gamma(value ca_in)
{
    return REAL(0.2) / (REAL(1.0) + exp((ca_in - REAL(0.75)) / REAL(0.8)));
}

PREFIX value fca_inf(value ca_in)
{
    return (fca_alpha(ca_in) + fca_beta(ca_in) + fca_gamma(ca_in) + REAL(0.23)) / REAL(1.46);
}

PREFIX INLINE value k_fca(value v, value ca_in, value fca)
{
    return fca_inf(ca_in)>fca ? (v>-REAL(60.0) ? REAL(0.0) : REAL(1.0)) : REAL(1.0);
}

PREFIX value b_inf(value v)
{
    return pow(REAL(1.0) + exp(-(v + REAL(37.49098)) / REAL(5.40634)),-REAL(1.0));
}

PREFIX INLINE value tau_b(value v)
{
    return REAL(0.6) + REAL(5.4) / (REAL(1.0) + exp(REAL(0.03) * (v + REAL(100.0))));
}

PREFIX value b_tau_er(value v)
{
    return exp(- TIMESTEP / tau_b(v));
}

PREFIX value g_inf(value v)
{
    return REAL(1.0) / (REAL(1.0) + exp((v + REAL(66.0)) / REAL(6.0)));
}

PREFIX INLINE value tau_g(value v)
{
    return REAL(1.0) + REAL(40.0) / (REAL(1.0) + exp(REAL(0.08) * (v + REAL(65.0))));
}

PREFIX value g_tau_er(value v)
{
    return exp(- TIMESTEP /  tau_g(v));
}

PREFIX value rev_potential_ca(value ca_in)
{
    return CONST(RTtoF) / REAL(2.0) * log(CA_OUT / ca_in);
}

PREFIX value G_KS(value ca_in)
{
    return REAL(1.0) * REAL(0.0866) * (REAL(1.0) + REAL(0.6) / (REAL(1.0) + pow(REAL(0.000038) / ca_in , REAL(1.4))));
}

PREFIX INLINE value g_cal_factor(value v)
{
    return REAL(4.0) *  v * (SQR(FARADEY) / (R_GAS * TEMPERATURE)) /
        (exp(REAL(2.0) * v * CONST(FtoRT)) - REAL(1.0));
}

PREFIX value g_cal_first(value v)
{
    return REAL(0.25) * exp(REAL(2.0) * (v - REAL(15.0)) * CONST(FtoRT)) *
        g_cal_factor(v);
}

PREFIX value g_cal_second(value v)
{
    return CA_OUT * g_cal_factor(v);
}

PREFIX value g_cal_first_abs(value v)
{
    return exp(REAL(2.0) * v * CONST(FtoRT)) * g_cal_factor(v);
}

PREFIX value g_cal_second_abs(value v)
{
    return REAL(0.341) * CA_OUT * g_cal_factor(v);
}

#undef INLINE
#undef PREFIX
#undef CONST

#endif /* FUNCTIONS_TCC */
