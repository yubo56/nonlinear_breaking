'''
2d code with navier stokes dissipation
'''
import numpy as np
import sys
from strat_helper import *
from mpi4py import MPI
CW = MPI.COMM_WORLD

H = 1
XMAX = 4 * H
ZMAX = 10 * H

NUM_SNAPSHOTS = 300
TARGET_DISP_RAT = 0.1 # k_z * u_z / omega at base

PARAMS_DEFAULT = {'XMAX': XMAX,
                  'ZMAX': ZMAX,
                  'N_X': 256,
                  'N_Z': 1024,

                  'KX': 2 * np.pi / XMAX,
                  'KZ': -2 * np.pi / H,
                  'g': H,
                  'H': H,
                  'RHO0': 1,
                  'Z0': 0.2 * ZMAX,
                  'Re_inv': 1,

                  'F_MULT': 1,
                  'SPONGE_STRENGTH': 15,
                  'SPONGE_WIDTH': 0.5,
                  'SPONGE_HIGH': 0.95 * ZMAX,
                  'SPONGE_LOW': 0.03 * ZMAX,

                  'NUM_SNAPSHOTS': NUM_SNAPSHOTS,
                  'NL': True,
                  'mask': False}

def get_params(overrides=None):
    params = {**PARAMS_DEFAULT, **(overrides or {})}
    g = params['g']
    KX = params['KX']
    KZ = params['KZ']
    OMEGA = get_omega(g, H, KX, KZ)
    VG_Z = get_vgz(g, H, KX, KZ)

    PARAMS_DEFAULT['T_F'] = abs(ZMAX / VG_Z) * 6
    PARAMS_DEFAULT['OMEGA'] = OMEGA
    PARAMS_DEFAULT['S'] = PARAMS_DEFAULT['ZMAX'] / 512 * 4
    PARAMS_DEFAULT['DT'] = min(0.1 / OMEGA, 0.1)

    # second override unfortunately
    params = {**PARAMS_DEFAULT, **(overrides or {})}

    params['F'] = params['F_MULT'] * \
        (TARGET_DISP_RAT * OMEGA / KZ) / get_uz_f_ratio(params) \
        * np.exp(-params['Z0'] / (2 * H))
    # NU / (kmax)^2 ~ omega
    params['NU'] = params['Re_inv'] * \
        OMEGA * params['ZMAX'] / (2 * np.pi * params['N_Z']) / abs(KZ)

    if CW.rank == 0: # print only on root process
        print(params)
    return params

if __name__ == '__main__':
    tasks = [
       # ('lin_1_masked',
       #  get_params(overrides={'F_MULT': 0.01,
       #                        'mask': True,
       #                        'NL': True,
       #                        'N_X': 64,
       #                        'N_Z': 256,
       #                        'Re_inv': 0.075})),
       # ('lin_2',
       #  get_params(overrides={'F_MULT': 0.0005,
       #                        'T_F': 4000,
       #                        'DT': 0.1,
       #                        'NL': False,
       #                        'N_X': 64,
       #                        'N_Z': 256,
       #                        'Re_inv': 0})),
       # ('nl_1_masked',
       #  get_params(overrides={'F_MULT': 1,
       #                        'mask': True,
       #                        'NL': True,
       #                        'Re_inv': 0.7})),
       ('nl_4_masked',
        get_params(overrides={'F_MULT': 1,
                              'mask': True,
                              'NL': True,
                              'Re_inv': 0.3})),
       # ('nl_6_masked',
       #  get_params(overrides={'F_MULT': 1,
       #                        'mask': True,
       #                        'Re_inv': 0.2})),
    ]
    if '-plot' in sys.argv:
        for name, params_dict in tasks:
            plot(name, params_dict)

    elif '-merge' in sys.argv:
        for name, _ in tasks:
            merge(name)

    elif '-write' in sys.argv:
        for name, params_dict in tasks:
            write_front(name, params_dict)

    elif '-front' in sys.argv:
        for name, params_dict in tasks:
            plot_front(name, params_dict)

    else:
        for task in tasks:
            run_strat_sim(*task)
