//                                                                   -*- C++ -*-
#include <cmath>
#include <cstdio>
#include <cassert>

#include "util.h"
#include "cuda.h"
#include "statevar.h"
#include "tabulation.cuh"
#include "tabulate.cuh"
#include "parameters.cuh"
#include "step.h"
#include "debug.h"

#include "functions.tcc"

__device__ real current_reaction(var_array_t *vars, int address_new, real voltij)
{
    real current;

    real index = TAB_INDEX(M_inf, voltij);
    real ca = GET_STRUCT_ADDRESS(vars, Ca, address_new);
    real ca_jsr = GET_STRUCT_ADDRESS(vars, Ca_JSR, address_new);
    real ca_nsr = GET_STRUCT_ADDRESS(vars, Ca_NSR, address_new);
    real fca = GET_STRUCT_ADDRESS(vars, FCa, address_new);
    real p_o1 = GET_STRUCT_ADDRESS(vars, P_o1, address_new);

    assert(TAB_WITHIN(H_inf, voltij));
    assert(TAB_WITHIN(Rev_potential_ca, ca));

//modif grad andrey 06.02.2022
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    /*int tj = threadIdx.x + 1;
    int ti = threadIdx.y + 1;
    int address_new = ADDRESS(i, j);
    real voltij = GET_VOLT_ADDRESS(volt, address_new);
    real index = TAB_INDEX(M_inf, voltij);
    real ca = GET_STRUCT_ADDRESS(vars, Ca, address_new);
    real ca_jsr = GET_STRUCT_ADDRESS(vars, Ca_JSR, address_new);
    real ca_nsr = GET_STRUCT_ADDRESS(vars, Ca_NSR, address_new);
    real fca = GET_STRUCT_ADDRESS(vars, FCa, address_new);
    real p_o1 = GET_STRUCT_ADDRESS(vars, P_o1, address_new);

    assert(TAB_WITHIN(H_inf, volt));
    assert(TAB_WITHIN(Rev_potential_ca, ca));*/

//modif fin

    real cur_na = G_NA*(REAL(3.25) - REAL(2.38)*j/DIMY) * CUBE(GET_STRUCT_ADDRESS(vars, M, address_new)) *
        GET_STRUCT_ADDRESS(vars, H, address_new) *
        GET_STRUCT_ADDRESS(vars, J, address_new) * (voltij - const_ENa);

    real cur_fNa = G_F * GET_STRUCT_ADDRESS(vars, Y, address_new) * 0.2*(voltij - const_ENa);
    real cur_fK = G_F * GET_STRUCT_ADDRESS(vars, Y, address_new) * 0.8 *(voltij - const_EK);
    real cur_f = (cur_fK + cur_fNa);

    /*real cur_k1 = (REAL(0.103) * (const_K_OUT/(const_K_OUT + REAL(210.0))) * (volt - const_EK - REAL(6.1373)) / (REAL(0.1653) 
    + exp(REAL(0.0319) * (volt - const_EK - REAL(6.1373)))));*/

    real cur_k1 = G_K1 * xk1_abs(voltij) * (voltij - const_EK);

    real cur_k_to = G_TRAN_OUT*(REAL(2.5) + REAL(1.25)*j/DIMY) * GET_STRUCT_ADDRESS(vars, R, address_new) * (REAL(0.706) * GET_STRUCT_ADDRESS(vars, S, address_new)
        + REAL(0.294) * GET_STRUCT_ADDRESS(vars, Sslow, address_new)) * (voltij - const_EK);

    real cur_ks =G_KS(ca) * (REAL(2.5) + REAL(1.25)*j/DIMY) * GET_STRUCT_ADDRESS(vars, Xs1, address_new) * GET_STRUCT_ADDRESS(vars, Xs2, address_new) * (voltij - const_EKs); 

    real Rr = REAL(1.0) / (REAL(1.0) + exp((voltij + REAL(9.0)) / REAL(22.4)));
    real cur_kr = const_G_KR*(REAL(2.5) + REAL(1.25)*j/DIMY) * GET_STRUCT_ADDRESS(vars, Xr, address_new) * Rr * (voltij - const_EK);

    real cur_ca_ltype = G_CaL*(REAL(0.76) - REAL(0.58)*j/DIMY)*(GET_STRUCT_ADDRESS(vars, D, address_new) * GET_STRUCT_ADDRESS(vars, F, address_new) * GET_STRUCT_ADDRESS(vars, FCa, address_new) *
        (TAB_VALUE(G_cal_first_abs, index) * ca - TAB_VALUE(G_cal_second_abs, index)));

    real cur_ca_ttype = G_CaT * GET_STRUCT_ADDRESS(vars, B, address_new) * GET_STRUCT_ADDRESS(vars, G, address_new) * 
    (voltij - rev_potential_ca(ca) + REAL(106.5));

    real NCX_c1 = CUBE(NA_IN) * CA_OUT * exp(REAL(0.03743) * GAMMA * voltij ) - CUBE(NA_OUT) * ca * 
    exp(REAL(0.03743) * (GAMMA - REAL(1.0)) * voltij);
    real NCX_c2 = REAL(1.0) + const_d_NCX * (CUBE(NA_OUT) * ca + CUBE(NA_IN) * CA_OUT);

    real cur_NCX = const_k_NCX * NCX_c1 / NCX_c2;

    real cur_ca_bg = G_CA_BACKGROUND * (voltij - rev_potential_ca(ca));

    real cur_na_bg = G_NA_BACKGROUND * (voltij - const_ENa);

    real cur_pump = const_pump_factor / (REAL(1.0) + REAL(0.1245) * exp(-REAL(0.1) * voltij * const_FtoRT) 
    + REAL(0.0365) * const_SIGMA * exp(-REAL(1.0) * voltij * const_FtoRT));

    real J_rel = NU1 * p_o1 * (ca_jsr - ca);
    real J_leak = KLEAK * (ca_nsr - ca);
    real J_tr = (ca_nsr - ca_jsr) / TAU_TR;
    real beta_SR = REAL(1.0) / (REAL(1.0) + CSQN_TOT * KMCSQN / SQR(ca_jsr + KMCSQN));

    real K_mRyR = REAL(3.51) / (REAL(1.0) + exp((ca_jsr - REAL(530.0)) / REAL(200.0))) + REAL(0.25);
    real P_C1 = REAL(1.0) - p_o1;

    real s1 = SQR(ca / REAL(0.5));
    real s2 = SQR(ca_nsr / REAL(3500.0));
    real J_up = REAL(0.9996)*(s1 - s2) / (REAL(1.0) + s1 + s2);
    real J_CaSR = J_rel-J_up+J_leak;
    real J_CaSL = (REAL(2.0) * cur_NCX - cur_ca_ltype - cur_ca_ttype - cur_ca_bg) * const_ACAP * CAPACITANCE / 
    (REAL(2.0) * FARADEY * pow(REAL(10.0),-REAL(6.0)));

    real beta_Cai = REAL(1.0) / (REAL(1.0) + (TRPN_TOT * KMTRPN / SQR(ca + KMTRPN)) + (CMDN_TOT * KMCMDN / SQR(ca + KMCMDN)));

    current = cur_ca_ltype + cur_ca_ttype + cur_NCX + cur_ca_bg + cur_na_bg + cur_pump + cur_na + cur_f
        + cur_k1 + cur_k_to + cur_ks + cur_kr;

    ca_nsr += TIMESTEP * (J_up - J_leak - J_tr) / const_V_NSR;
    ca_jsr += TIMESTEP * beta_SR * (-J_rel + J_tr) / const_V_JSR;

    p_o1 += TIMESTEP * (k_a_plus * pow(ca, REAL(4.0)) / (pow(ca, REAL(4.0)) + pow(K_mRyR, REAL(4.0))) * P_C1 - k_a_minus * p_o1);

    ca += TIMESTEP * beta_Cai * (J_CaSR+J_CaSL) / const_V_MYO;
 
    fca += TIMESTEP * k_fca(voltij, ca, fca) * (fca_inf(ca) - fca) / TAU_FCA;

#define UPDATE(A, a)                            \
     real a##_inf_val = a##_inf(voltij);                  \
    SET_STRUCT_ADDRESS(vars, A, address_new, a##_inf_val -     \
        (a##_inf_val - GET_STRUCT_ADDRESS(vars, A, address_new)) * \
            TAB_VALUE(A##_tau_er, index))

    UPDATE(M, m);
    UPDATE(H, h);
    UPDATE(J, j);
    UPDATE(D, d);
    UPDATE(F, f);
    UPDATE(B, b);
    UPDATE(G, g);
    UPDATE(Y, y);
    UPDATE(R, r);
    UPDATE(S, s);
    UPDATE(Sslow, sslow);
    UPDATE(Xr, xr);
    UPDATE(Xs1, xs1);
    UPDATE(Xs2, xs2);

#undef UPDATE

    SET_STRUCT_ADDRESS(vars, FCa, address_new, fca);
    SET_STRUCT_ADDRESS(vars, P_o1, address_new, p_o1);
    SET_STRUCT_ADDRESS(vars, Ca_JSR, address_new, ca_jsr);
    SET_STRUCT_ADDRESS(vars, Ca_NSR, address_new, ca_nsr);
    SET_STRUCT_ADDRESS(vars, Ca, address_new, ca);

    return current;
}


__global__ void step_kernel(voltage_t volt, voltage_t volt_new,
                var_array_t *vars, weights_array_t *weights)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int tj = threadIdx.x + 1;
    int ti = threadIdx.y + 1;
    int address = ADDRESS(i, j);
    real voltij = GET_VOLT_ADDRESS(volt, address);

    __shared__ real tile[BLOCK_DIMY + 2][BLOCK_DIMX + 2];

    if (threadIdx.x == 0) {
        
        if (threadIdx.y == 0)
            tile[0][0] = GET_VOLT(volt, i - 1, j - 1);    
        if (threadIdx.y == BLOCK_DIMY - 1)
            tile[BLOCK_DIMY + 1][0] = GET_VOLT(volt, i + 1, j - 1);

        tile[ti][0] = GET_VOLT(volt, i, j - 1);
    }

    if (threadIdx.x == BLOCK_DIMX - 1) {

        if (threadIdx.y == 0)
            tile[0][BLOCK_DIMX + 1] = GET_VOLT(volt, i - 1, j + 1);    
        if (threadIdx.y == BLOCK_DIMY - 1)
            tile[BLOCK_DIMY + 1][BLOCK_DIMX + 1] = GET_VOLT(volt, i + 1, j + 1);

        tile[ti][BLOCK_DIMX + 1] = GET_VOLT(volt, i, j + 1);
    }

    if (threadIdx.y == 0)
        tile[0][tj] = GET_VOLT(volt, i - 1, j);
    if (threadIdx.y == BLOCK_DIMY - 1)
        tile[BLOCK_DIMY + 1][tj] = GET_VOLT(volt, i + 1, j);
    

    tile[ti][tj] = voltij;
    __syncthreads();

    real current = current_reaction(vars, address, voltij);

    real flux_top_left_i = (tile[ti][tj - 1] - tile[ti - 1][tj - 1]) + (tile[ti][tj] - tile[ti - 1][tj]);
    real flux_top_left_j = (tile[ti - 1][tj] - tile[ti - 1][tj - 1]) + (tile[ti][tj] - tile[ti][tj - 1]);
    real flux_top_right_i = (tile[ti][tj] - tile[ti - 1][tj]) + (tile[ti][tj + 1] - tile[ti - 1][tj + 1]);
    real flux_top_right_j = (tile[ti - 1][tj + 1] - tile[ti - 1][tj]) + (tile[ti][tj + 1] - tile[ti][tj]);

    real flux_bottom_left_i = (tile[ti + 1][tj - 1] - tile[ti][tj - 1]) + (tile[ti + 1][tj] - tile[ti][tj]);
    real flux_bottom_left_j = (tile[ti][tj] - tile[ti][tj - 1]) + (tile[ti + 1][tj] - tile[ti + 1][tj - 1]);
    real flux_bottom_right_i = (tile[ti + 1][tj] - tile[ti][tj]) + (tile[ti + 1][tj + 1] - tile[ti][tj + 1]);
    real flux_bottom_right_j = (tile[ti][tj + 1] - tile[ti][tj]) + (tile[ti + 1][tj + 1] - tile[ti + 1][tj]);

    real cur_ax = const_Diff_to_Step_2 *
        (GET_STRUCT_ADDRESS(weights, next_i, address) * (tile[ti + 1][tj] - voltij) +
         GET_STRUCT_ADDRESS(weights, prev_i, address) * (tile[ti - 1][tj] - voltij) +
         GET_STRUCT_ADDRESS(weights, next_j, address) * (tile[ti][tj + 1] - voltij) +
         GET_STRUCT_ADDRESS(weights, prev_j, address) * (tile[ti][tj - 1] - voltij)) +
    const_Diff_to_Step_8 * (
        GET_STRUCT_ADDRESS(weights, dxy_top_right, address) * flux_top_right_i -
        GET_STRUCT_ADDRESS(weights, dxy_top_left, address) * flux_top_left_i +
        GET_STRUCT_ADDRESS(weights, dxy_bottom_right, address) * flux_bottom_right_i -
        GET_STRUCT_ADDRESS(weights, dxy_bottom_left, address) * flux_bottom_left_i +
        GET_STRUCT_ADDRESS(weights, dyx_bottom_left, address) * flux_bottom_left_j -
        GET_STRUCT_ADDRESS(weights, dyx_top_left, address) * flux_top_left_j +
        GET_STRUCT_ADDRESS(weights, dyx_bottom_right, address) * flux_bottom_right_j -
        GET_STRUCT_ADDRESS(weights, dyx_top_right, address) * flux_top_right_j);

    SET_VOLT_ADDRESS(volt_new, address, voltij + TIMESTEP * (cur_ax - current));
}

extern "C"
void step(voltage_t volt, voltage_t volt_new, var_array_t *vars,
      weights_array_t *weights)
{
    dim3 block(BLOCK_DIMX, BLOCK_DIMY);
    dim3 grid(DIMX / BLOCK_DIMX, DIMY / BLOCK_DIMY);

    step_kernel <<<grid, block, 0, Cuda_Stream[COMPUTATION]>>>
    (volt, volt_new, vars, weights);
}
