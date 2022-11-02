#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#ifdef __cplusplus
extern "C" {
#endif

double m_inf(double v);
double m_tau_er(double v);
double h_inf(double v);
double h_tau_er(double v);
double j_inf(double v);
double j_tau_er(double v);
double y_inf(double v);
double y_tau_er(double v);
double b_inf(double v);
double b_tau_er(double v);
double g_inf(double v);
double g_tau_er(double v);
double xr_inf(double v);
double xr_tau_er(double v);
double xs1_inf(double v);
double xs1_tau_er(double v);
double xs2_inf(double v);
double xs2_tau_er(double v);
double xk1_abs(double v);
double r_inf(double v);
double r_tau_er(double v);
double s_inf(double v);
double s_tau_er(double v);
double sslow_inf(double v);
double sslow_tau_er(double v);
double d_inf(double v);
double d_tau_er(double v);
double f_inf(double v);
double f_tau_er(double v);
double fca_inf(double v);
double rev_potential_ca(double ca);
double G_KS(double ca);
double g_cal_first(double v);
double g_cal_second(double v);
double g_cal_first_abs(double v);
double g_cal_second_abs(double v);

#ifdef __cplusplus
}
#endif

#endif	/* FUNCTIONS_H */
