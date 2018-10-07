'''
TODO too much weird stuff happens atop the critical layer, what to do?
'''
import numpy as np
import sys
from strat_helper import *

H = 1e3
XMAX = H
ZMAX = H

NUM_SNAPSHOTS = 100
TARGET_DISP_RAT = 0.005 # k_z * u_z / omega at base

PARAMS_RAW = {'XMAX': XMAX,
              'ZMAX': ZMAX,
              'N_X': 512,
              'N_Z': 512,
              'KX': 2 * np.pi / H,
              'KZ': -12 * np.pi / H,
              'H': H,
              'RHO0': 1,
              'Z0': 0.15 * ZMAX,
              'SPONGE_STRENGTH': 2,
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
    T_F = abs(ZMAX / VG_Z) * 3

    params['T_F'] = T_F
    params['g'] = g
    params['OMEGA'] = OMEGA
    params['S'] = params['ZMAX'] / 512 * 6
    params['INTERP_X'] = interp_x
    params['INTERP_Z'] = interp_z
    params['N_X'] //= interp_x
    params['N_Z'] //= interp_z
    params['DT'] = min(0.1 / OMEGA, 0.05)
    params['F'] = params.get('F_MULT', 0.1) * \
        (TARGET_DISP_RAT * OMEGA / KZ) / get_uz_f_ratio(params) \
        * np.exp(-params['Z0'] / (2 * H))
    # for nabla^n visc, u / (nu * kx^{n-1}) = 1
    params['NU_X'] = params.get('NU_MULT', 1) * \
        OMEGA * (params['XMAX'] / (2 * np.pi * params['N_X']))**5 / abs(KZ)
    params['NU_Z'] = params.get('NU_MULT', 1) * \
        OMEGA * (params['ZMAX'] / (2 * np.pi * params['N_Z']))**5 / abs(KZ)

    # Ri = 1/4 gives instability + weird reflection
    # params['DUZ_DZ'] = np.sqrt((g / H) / params.get('Ri', 1/8))
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
        (zero_ic, 'vstrat',
         build_interp_params(1, 1)),
        (zero_ic, 'vstrat2',
         build_interp_params(1, 1, overrides={'NU_MULT': 8})),
    ]
    if '-plot' not in sys.argv:
        for task in tasks:
            run(*task)

    else:
        for _, name, params_dict in tasks:
            plot(name, params_dict)
