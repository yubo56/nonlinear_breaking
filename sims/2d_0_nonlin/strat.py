'''
2d_1_strat + Navier-Stokes dissipation
'''
import numpy as np
import sys
from strat_helper import *

H = 1
XMAX = 4 * H
ZMAX = 10 * H
KX = 2 * np.pi / XMAX
KZ = -20 / H
g = (KX**2 + KZ**2 + 1 / (4 * H**2)) / KX**2 * (2 * np.pi)**2 * H # omega = 2pi
OMEGA = get_omega(g, H, KX, KZ)

num_timesteps = 2e3
NUM_SNAPSHOTS = 200

PARAMS_RAW = {'XMAX': XMAX,
              'ZMAX': ZMAX,
              'N_X': 256,
              'N_Z': 1024,
              'OMEGA': OMEGA,
              'KX': KX,
              'KZ': KZ,
              'H': H,
              'RHO0': 1,
              'g': g,
              'A': 0.005,
              'F': 0.002,
              'S': abs(1 / KZ),
              'Z0': 0.15 * ZMAX,
              'SPONGE_STRENGTH': 2,
              'SPONGE_HIGH': 0.9 * ZMAX,
              'SPONGE_LOW': 0.1 * ZMAX,
              'NUM_SNAPSHOTS': NUM_SNAPSHOTS}

def build_interp_params(interp_x, interp_z, overrides=None):
    params = {**PARAMS_RAW, **(overrides or {})}

    _, VG_Z = get_vg(g, H, params['KX'], params['KZ'])
    T_F = abs(ZMAX / VG_Z) * 1.2

    params['T_F'] = T_F
    params['INTERP_X'] = interp_x
    params['INTERP_Z'] = interp_z
    params['N_X'] //= interp_x
    params['N_Z'] //= interp_z
    params['DT'] = params['T_F'] / num_timesteps
    params['NU'] = 0.1 * (ZMAX / params['N_Z'])**2 / np.pi**2 # smallest wavenumber
    return params

def run(ic, name, params_dict):
    assert 'INTERP_X' in params_dict and 'INTERP_Z' in params_dict,\
        'params need INTERP'
    run_strat_sim(ic, name, params_dict)
    return '%s completed' % name

if __name__ == '__main__':
    tasks = [
        (zero_ic, 'linear_1',
         build_interp_params(1, 1, overrides={'F': 0.00001})),
        (zero_ic, 'linear_2',
         build_interp_params(1, 1, overrides={'KX': 2 * np.pi / H, 'F':
                                              0.00001})),
        (zero_ic, 'linear_3',
         build_interp_params(1, 1, overrides={'KX': 16 * np.pi / H, 'F':
                                              0.00001})),
        (zero_ic, 'nonlinear_1',
         build_interp_params(1, 1, overrides={'F': 0.000001})),
        (zero_ic, 'nonlinear_2',
         build_interp_params(1, 1, overrides={'KX': 2 * np.pi / H, 'F':
                                              0.00001})),
        (zero_ic, 'nonlinear_3',
         build_interp_params(1, 1, overrides={'KX': 16 * np.pi / H, 'F':
                                              0.00001})),
    ]
    if '-plot' not in sys.argv:
        for task in tasks:
            run(*task)

    else:
        for _, name, params_dict in tasks:
            plot(name, params_dict)
