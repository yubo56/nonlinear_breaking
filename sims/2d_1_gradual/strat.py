'''
2d code with navier stokes dissipation
'''
import numpy as np
import sys
from strat_helper import *

H = 1
XMAX = 4 * H
ZMAX = 12 * H

NUM_SNAPSHOTS = 300
TARGET_DISP_RAT = 0.2 # k_z * u_z / omega at base

PARAMS_RAW = {'XMAX': XMAX,
              'ZMAX': ZMAX,
              'N_X': 256,
              'N_Z': 512,
              'KX': 2 * np.pi / XMAX,
              'KZ': -20 / H,
              'H': H,
              'RHO0': 1,
              'Z0': 0.15 * ZMAX,
              'SPONGE_STRENGTH': 0.6,
              'SPONGE_WIDTH': 0.5,
              'SPONGE_HIGH': 0.93 * ZMAX,
              'SPONGE_LOW': 0.07 * ZMAX,
              'NUM_SNAPSHOTS': NUM_SNAPSHOTS}

def build_interp_params(interp_x, interp_z, overrides=None):
    params = {**PARAMS_RAW, **(overrides or {})}
    KX = params['KX']
    KZ = params['KZ']
    g = H # N^2 = 1

    OMEGA = get_omega(g, H, KX, KZ)
    VG_Z = get_vgz(g, H, KX, KZ)
    T_F = abs(ZMAX / VG_Z) * 1.2

    params['T_F'] = T_F
    params['g'] = g
    params['OMEGA'] = OMEGA
    params['S'] = abs(1 / KZ)
    params['INTERP_X'] = interp_x
    params['INTERP_Z'] = interp_z
    params['N_X'] //= interp_x
    params['N_Z'] //= interp_z
    # omega * DT << 1 is required, as is DT << 1/N = 1
    params['DT'] = min(0.1 / OMEGA, 0.5)
    if not params.get('F'): # default value
        params['F'] = (TARGET_DISP_RAT * OMEGA / KZ) / get_uz_f_ratio(params) \
            * np.exp(-params['Z0'] / (2 * H))
    # NU / (kmax)^2 ~ omega
    params['NU_X'] = params.get('NU_MULT_X', 0.5) * \
        OMEGA * (params['XMAX'] / (2 * np.pi * params['N_X']))**2
    params['NU_Z'] = params.get('NU_MULT_Z', 0.5) * \
        OMEGA * (params['ZMAX'] / (2 * np.pi * params['N_Z']))**2
    print(params)
    return params

def run(ic, name, params_dict):
    try:
        run_strat_sim(ic, name, params_dict)
    except FloatingPointError as e:
        print(e)
        pass

if __name__ == '__main__':
    tasks = [
        # (zero_ic, 'nonlinear_ns_lowres',
        #  build_interp_params(2, 1, overrides={'USE_CFL': True})),
        # (zero_ic, 'nonlinear_ns',
        #  build_interp_params(1, 1, overrides={'USE_CFL': True})),

        (zero_ic, 'nonlinear_ns_lowres1',
         build_interp_params(1, 1, overrides={'USE_CFL': True,
                                              'NU_MULT_X': 1})),
        (zero_ic, 'nonlinear_ns_lowres2',
         build_interp_params(1, 1, overrides={'USE_CFL': True,
                                              'NU_MULT_X': 2})),
        (zero_ic, 'nonlinear_ns_lowres3',
         build_interp_params(1, 1, overrides={'USE_CFL': True,
                                              'NU_MULT_X': 1,
                                              'NU_MULT_Z': 1/8})),
        (zero_ic, 'nonlinear_ns_lowres4',
         build_interp_params(1, 1, overrides={'USE_CFL': True,
                                              'NU_MULT_X': 4})),
    ]
    if '-plot' not in sys.argv:
        for task in tasks:
            run(*task)

    else:
        for _, name, params_dict in tasks:
            plot(name, params_dict)
