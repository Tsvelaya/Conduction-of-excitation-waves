/*                                                           -*- C -*-       */
#ifndef PARAMETERS_CUH
#define PARAMETERS_CUH

#include "value.h"

/* Cell Type */
#define EPI

/* Fundamental constants */

#define R_GAS REAL(8.314)
#define FARADEY REAL(96.5)
#define TEMPERATURE REAL(305.0)
#define CAPACITANCE REAL(1.0)
#define CAP_FIBRO REAL(0.0000063)
#define r_nucleus REAL(5.7934)             
#define r_SR REAL(6.0)                        
#define r_SL REAL(10.5)
#define RADIUS REAL(225.0)

/* INTEGRATION STEPS */
/* timestep (ms) */
#define TIMESTEP REAL(0.002)
#define SPACESTEP REAL(0.03125)

#define DIFFUSION_S REAL(0.23)		/* 0.23 -- 0.91 */
#define DIFFUSION_RHO REAL(1.75)
#define SURFACE REAL(0.01)

#define G_GAP REAL(0.5)
#define G_FIBRO REAL(0.0)
#define DIFFUSION_M REAL(0.025)		/* Normal: 0.025 */
#define DIFFUSION_F REAL(0.00625)		/* Normal: 0.00625 */
#define REDUCTION_FACTOR REAL(1.0)
/* Concentrations */
//#define K_OUT REAL(5366.0)
#define CA_OUT REAL(1796.0)
#define NA_OUT REAL(154578.0)

/* Other constants */
#define INAK_MAX REAL(3.1993)
#define GAMMA REAL(0.5)
#define KMNAI REAL(18600.0)
#define nNAK REAL(3.2)
#define KMKO REAL(1500.0)
#define P_NaK REAL(0.01833)

/* CURRENTS */
#define G_TRAN_OUT REAL(0.1)
#define G_F REAL(0.021)
#define G_NA REAL(20.0)
#define G_K1 REAL(5.4) * REAL(0.58)
#define G_NA_BACKGROUND REAL(0.0039)
#define G_CaL REAL(0.0000567)
#define G_CaT REAL(0.2)
#define G_CA_BACKGROUND REAL(0.0008)

#define NU1 REAL(0.01)
#define k_a_plus REAL(1.0)
#define k_a_minus REAL(0.16)

#define KLEAK REAL(0.000005)
#define CSQN_TOT REAL(24750.0)
#define KMCSQN REAL(800.0)
#define TRPN_TOT REAL(35.0)
#define KMTRPN REAL(0.5)
#define CMDN_TOT REAL(50.0)
#define KMCMDN REAL(2.38)

/* INITIAL CONDITIONS */
#define VOLTAGE_INIT REAL(-72.79684)
#define VOLTAGE_FIBRO_INIT REAL(-20.0)
#define CA_IN_INIT REAL(0.22515)
#define NA_IN REAL(13818.5982638)
#define K_IN REAL(150953.3914836)
#define TAU_FCA REAL(10.0)
#define TAU_TR REAL(200.0)

#ifdef __CUDACC__
#define DECLARE_CONSTANT(a) extern real a; extern __constant__ real const_##a
#else
#define DECLARE_CONSTANT(a) extern real a
#endif
#include "constants.h"
#undef DECLARE_CONSTANT

#ifdef __cplusplus
extern "C"
#endif
void init_constants();

/* TABULATION */

#define TAB_VOLTAGE_MIN REAL(-100.0)
#define TAB_VOLTAGE_MAX REAL(80.0)
#define TAB_CA_MIN REAL(0.0)
#define TAB_CA_MAX REAL(2.0)
#define TAB_POINTS 10000

/* F_GATE_REDUCTION */
#define F_GATE_RED REAL(1.0)

/* PACING_CYCLE_STEP*/
#define PACING_CYCLE REAL(300.0) /*ms*/
#define XLENGTH 100
#define YLENGTH 200

#define LINE 0
#define THR_FRONT 1
#define EDGE 0

#endif	/* PARAMETERS_CUH */
