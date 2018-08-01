'''
2d code with navier stokes dissipation
'''
import numpy as np
import sys
from strat_helper import *

H = 1
XMAX = 4 * H
ZMAX = 10 * H

NUM_TIMESTEPS = 2e3
NUM_SNAPSHOTS = 200
TARGET_UZ = 0.01 # target uz at forcing zone

PARAMS_RAW = {'XMAX': XMAX,
              'ZMAX': ZMAX,
              'N_X': 128,
              'N_Z': 1024,
              'KX': 2 * np.pi / XMAX,
              'KZ': -20 / H,
              'H': H,
              'RHO0': 1,
              'Z0': 0.15 * ZMAX,
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
    T_F = abs(ZMAX / VG_Z) * 1.2

    params['T_F'] = T_F
    params['g'] = g
    params['OMEGA'] = OMEGA
    params['S'] = abs(1 / KZ)
    params['INTERP_X'] = interp_x
    params['INTERP_Z'] = interp_z
    params['N_X'] //= interp_x
    params['N_Z'] //= interp_z
    # omega * DT << 1 is required
    params['DT'] = min(params['T_F'] / NUM_TIMESTEPS, 0.01 / OMEGA)
    if not params.get('F'): # default value
        params['F'] = TARGET_UZ / get_uz_f_ratio(params)
    # NU / (kmax/2)^2 ~ omega
    params['NU'] = OMEGA * (params['ZMAX'] / (np.pi * params['N_Z']))**2
    print(params['NU'])
    return params

def run(ic, name, params_dict):
    try:
        run_strat_sim(ic, name, params_dict)
    except FloatingPointError as e:
        print(e)
        pass

if __name__ == '__main__':
    tasks = [
        (zero_ic, 'linear_ns',
         build_interp_params(1, 1, overrides={'F': 0.00001})),
        (zero_ic, 'nonlinear_ns',
         build_interp_params(1, 1)),
    ]
    if '-plot' not in sys.argv:
        for task in tasks:
            run(*task)

    else:
        for _, name, params_dict in tasks:
            plot(name, params_dict)
