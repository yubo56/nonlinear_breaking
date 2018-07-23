'''
2d_1_strat + Navier-Stokes dissipation
'''
import numpy as np
import sys
from strat_helper import *

H = 1
XMAX = 4 * H
ZMAX = 10 * H

num_timesteps = 2e3
NUM_SNAPSHOTS = 200

PARAMS_RAW = {'XMAX': XMAX,
              'ZMAX': ZMAX,
              'N_X': 256,
              'N_Z': 1024,
              'KX': 2 * np.pi / XMAX,
              'KZ': -20 / H,
              'H': H,
              'RHO0': 1,
              'A': 0.005,
              'F': 0.002,
              'Z0': 0.15 * ZMAX,
              'SPONGE_STRENGTH': 2,
              'SPONGE_HIGH': 0.9 * ZMAX,
              'SPONGE_LOW': 0.1 * ZMAX,
              'NUM_SNAPSHOTS': NUM_SNAPSHOTS}

def build_interp_params(interp_x, interp_z, overrides=None):
    params = {**PARAMS_RAW, **(overrides or {})}
    KX = params['KX']
    KZ = params['KZ']
    g = (KX**2 + KZ**2 + 1 / (4 * H**2)) / KX**2 * (2 * np.pi)**2 * H # omega = 2pi

    OMEGA = get_omega(g, H, KX, KZ)
    _, VG_Z = get_vg(g, H, KX, KZ)
    T_F = abs(ZMAX / VG_Z) * 1.2

    params['T_F'] = T_F
    params['g'] = g
    params['OMEGA'] = OMEGA
    params['S'] = abs(1 / KZ)
    params['INTERP_X'] = interp_x
    params['INTERP_Z'] = interp_z
    params['N_X'] //= interp_x
    params['N_Z'] //= interp_z
    params['DT'] = params['T_F'] / num_timesteps
    params['NU'] = 0.1 * (ZMAX / params['N_Z'])**2 / np.pi**2 # smallest wavenumber
    return params

def run(ic, name, params_dict):
    run_strat_sim(ic, name, params_dict)

if __name__ == '__main__':
    tasks = [
        (zero_ic, 'nonlinear_1',
         build_interp_params(1, 1)),
        (zero_ic, 'nonlinear_2',
         build_interp_params(1, 1, overrides={'KX': 4 * np.pi / H})),
        (zero_ic, 'nonlinear_3',
         build_interp_params(1, 1, overrides={'KX': 16 * np.pi / H})),

        (zero_ic, 'linear_1',
         build_interp_params(1, 1, overrides={'F': 0.00001})),
        (zero_ic, 'linear_2',
         build_interp_params(1, 1, overrides={'KX': 2 * np.pi / H,
                                              'F': 0.00001})),
        (zero_ic, 'linear_3',
         build_interp_params(1, 1, overrides={'KX': 16 * np.pi / H,
                                              'F': 0.00001})),
    ]
    if '-plot' not in sys.argv:
        for task in tasks:
            run(*task)

    else:
        for _, name, params_dict in tasks:
            plot(name, params_dict)
