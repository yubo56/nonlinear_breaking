'''
2d code with navier stokes dissipation
'''
import numpy as np
import sys
from strat_helper import *

H = 1
XMAX = 4 * H
ZMAX = 5 * H

NUM_TIMESTEPS = 2e3
NUM_SNAPSHOTS = 200
TARGET_DISP_RAT = 0.1 # k_z * u_z / omega at base

PARAMS_RAW = {'XMAX': XMAX,
              'ZMAX': ZMAX,
              'N_X': 64,
              'N_Z': 256,
              'KX': 2 * np.pi / XMAX,
              'KZ': -20 / H,
              'H': H,
              'RHO0': 1,
              'Z0': 0.2 * ZMAX,
              'SPONGE_STRENGTH': 1,
              'SPONGE_HIGH': 0.9 * ZMAX,
              'SPONGE_LOW': 0.1 * ZMAX,
              'NUM_SNAPSHOTS': NUM_SNAPSHOTS}

def build_interp_params(interp_x, interp_z, overrides=None):
    params = {**PARAMS_RAW, **(overrides or {})}
    KX = params['KX']
    KZ = params['KZ']
    # g = (KX**2 + KZ**2 + 1 / (4 * H**2)) / KX**2 * (2 * np.pi)**2 * H # omega = 2pi
    g = H # N^2 = 1

    OMEGA = get_omega(g, H, KX, KZ)
    VG_Z = get_vgz(g, H, KX, KZ)
    T_F = abs(ZMAX / VG_Z) * 2.5

    params['T_F'] = T_F
    params['g'] = g
    params['OMEGA'] = OMEGA
    params['S'] = abs(1 / KZ)
    params['INTERP_X'] = interp_x
    params['INTERP_Z'] = interp_z
    params['N_X'] //= interp_x
    params['N_Z'] //= interp_z
    # omega * DT << 1 is required, as is DT << 1/N = 1
    params['DT'] = min(params['T_F'] / NUM_TIMESTEPS, 0.1 / OMEGA, 0.2)
    if not params.get('F'): # default value
        params['F'] = (TARGET_DISP_RAT * OMEGA / KZ) / get_uz_f_ratio(params)
    # NU / (kmax/2)^2 ~ omega
    params['NU'] = params.get('NU_MULT', 1) * \
        OMEGA * (params['ZMAX'] / (np.pi * params['N_Z']))**2
    params['UZ_STRAT'] = params.get('UZ_STRAT_MULT', 1.2) * OMEGA / (KX * ZMAX)
    print(params)
    return params

def run(ic, name, params_dict):
    try:
        run_strat_sim(ic, name, params_dict)
    except Exception as e:
        print(e)
        pass

if __name__ == '__main__':
    tasks = [
        (zero_ic, 'vstrat_lin',
         build_interp_params(1, 1, overrides={'F': 1e-7, 'NU_MULT': 1e-4})),
        (zero_ic, 'vstrat_lownu',
         build_interp_params(1, 1, overrides={'NU_MULT': 1e-4})),
        (zero_ic, 'vstrat_highbreak',
         build_interp_params(1, 1, overrides={'UZ_STRAT_MULT': 0.8})),
        (zero_ic, 'vstrat',
         build_interp_params(1, 1)),
        (zero_ic, 'vstrat_highnu',
         build_interp_params(1, 1, overrides={'NU_MULT': 8})),
    ]
    if '-plot' not in sys.argv:
        for task in tasks:
            run(*task)

    else:
        for _, name, params_dict in tasks:
            plot(name, params_dict)
