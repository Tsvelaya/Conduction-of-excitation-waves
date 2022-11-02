//                                                                   -*- C++ -*-
#include <cmath>

#include "value.h"
#include "util.h"
#include "debug.h"
#include "options.h"

// constants.h should be included BEFORE parameters.cuh

#define DECLARE_CONSTANT(a) real a; __constant__ real const_##a
#include "constants.h"
#undef DECLARE_CONSTANT

// Do not forget to initiate all of the constants in constants.h

#include "parameters.cuh"

#define INIT(sym) cudaMemcpyToSymbol(const_##sym, &sym, sizeof(sym))

void init_constants()
{
    RTtoF = R_GAS * TEMPERATURE / FARADEY;
    INIT(RTtoF);

    FtoRT = FARADEY / (R_GAS * TEMPERATURE);
    INIT(FtoRT);

    Decrease = dec;
    INIT(Decrease);

    K_OUT = REAL(5366.0) * Decrease;
    INIT(K_OUT);

    G_KR = REAL(1.0) * REAL(0.0005228) * sqrt(K_OUT / REAL(5.4)); /* There is sqrt (1000) */
    INIT(G_KR);

    d_NCX = pow(REAL(10.0), -REAL(16.0));
    INIT(d_NCX);

    k_NCX = REAL(1.1340) * pow(REAL(10.0), -REAL(16.0));
    INIT(k_NCX);

    SIGMA = (exp(NA_OUT / REAL(67300.0)) - REAL(1.0))/ REAL(7.0);
    INIT(SIGMA);

    pump_factor = INAK_MAX / ((REAL(1.0) + pow(KMNAI/NA_IN, nNAK)) * (REAL(1.0) + KMKO/K_OUT));
    INIT(pump_factor);

    ENa = (R_GAS * TEMPERATURE / FARADEY) * log( NA_OUT / NA_IN );
    INIT(ENa);

    EK = (R_GAS * TEMPERATURE / FARADEY) * log( K_OUT / K_IN );
    INIT(EK);

    EKs = (R_GAS * TEMPERATURE / FARADEY) * log((K_OUT + P_NaK * NA_OUT) / (K_IN + P_NaK * NA_IN));;
    INIT(EKs);

    ACAP = REAL(4.0) * REAL(3.14159265) * SQR(r_SL) * pow(REAL(10.0),-REAL(8.0));
    INIT(ACAP);

    V_MYO = REAL(4.0) * REAL(3.14159265) * (CUBE(r_SL) - CUBE(r_SR)) / REAL(3000.0);
    INIT(V_MYO);

    V_NSR = REAL(0.08127);
    INIT(V_NSR);

    V_JSR = REAL(0.00903);
    INIT(V_JSR);

    Spacestep_2 = SQR(SPACESTEP);
    INIT(Spacestep_2);

    Diff_to_Step_2 = DIFFUSION_M / SQR(SPACESTEP);

    INIT(Diff_to_Step_2);

    Diff_to_Step_8 = Diff_to_Step_2 / 4;

    INIT(Diff_to_Step_8);
}

#undef INIT
