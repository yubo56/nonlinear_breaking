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

PARAMS_RAW = {'XMAX': XMAX,
              'ZMAX': ZMAX,
              'N_X': 256,
              'N_Z': 1024,
              'KX': 2 * np.pi / XMAX,
              'KZ': -2 * np.pi / H,
              'H': H,
              'RHO0': 1,
              'Z0': 0.2 * ZMAX,
              'SPONGE_STRENGTH': 15,
              'SPONGE_WIDTH': 0.5,
              'SPONGE_HIGH': 0.95 * ZMAX,
              'SPONGE_LOW': 0.03 * ZMAX,
              'NUM_SNAPSHOTS': NUM_SNAPSHOTS}

def build_interp_params(interp_x, interp_z, overrides=None):
    params = {**PARAMS_RAW, **(overrides or {})}
    KX = params['KX']
    KZ = params['KZ']
    g = H # N^2 = 1

    OMEGA = get_omega(g, H, KX, KZ)
    VG_Z = get_vgz(g, H, KX, KZ)
    T_F = abs(ZMAX / VG_Z) * 6

    params['T_F'] = params.get('T_F', T_F)
    params['NL'] = params.get('NL', True)
    params['adv_mask'] = params.get('adv_mask', False)
    params['g'] = g
    params['OMEGA'] = OMEGA
    params['S'] = params['ZMAX'] / 512 * 4
    params['INTERP_X'] = interp_x
    params['INTERP_Z'] = interp_z
    params['N_X'] //= interp_x
    params['N_Z'] //= interp_z
    # omega * DT << 1 is required, as is DT << 1/N = 1
    params['DT'] = params.get('DT', min(0.1 / OMEGA, 0.1))
    params['F'] = params.get('F_MULT', 1) * \
        (TARGET_DISP_RAT * OMEGA / KZ) / get_uz_f_ratio(params) \
        * np.exp(-params['Z0'] / (2 * H))
    # NU / (kmax)^2 ~ omega
    params['NU'] = params.get('Re', 1) * \
        OMEGA * params['ZMAX'] / (2 * np.pi * params['N_Z']) / abs(KZ)
    if CW.rank == 0: # print only on root process
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
       (set_ic, 'lin_1',
        build_interp_params(4, 4, overrides={'F_MULT': 0.01,
                                             'Re': 0.075})),
       (set_ic, 'lin_2',
        build_interp_params(4, 4, overrides={'F_MULT': 0.0005,
                                             'T_F': 10000,
                                             'DT': 0.1,
                                             'NL': False,
                                             'Re': 0})),
       (set_ic, 'nl_low_1',
        build_interp_params(2, 2, overrides={'F_MULT': 1,
                                             'Re': 0.5})),
       (set_ic, 'nl_1',
        build_interp_params(1, 1, overrides={'F_MULT': 1,
                                             'Re': 0.7})),
       (set_ic, 'nl_2',
        build_interp_params(1, 1, overrides={'F_MULT': 1,
                                             'Re': 0.4})),
       (set_ic, 'nl_3',
        build_interp_params(1, 1, overrides={'F_MULT': 2,
                                             'Re': 0.4})),
       (set_ic, 'nl_4',
        build_interp_params(1, 1, overrides={'F_MULT': 1,
                                             'Re': 0.3})),
       (set_ic, 'nl_4_masked',
        build_interp_params(1, 1, overrides={'F_MULT': 1,
                                             'adv_mask': True,
                                             'Re': 0.3})),
       (set_ic, 'nl_5',
        build_interp_params(1, 1, overrides={'F_MULT': 1,
                                             'Re': 0.3,
                                             'KZ': -4 * np.pi / H})),
       (set_ic, 'nl_6',
        build_interp_params(1, 1, overrides={'F_MULT': 1,
                                             'Re': 0.2})),
       (set_ic, 'nl_7',
        build_interp_params(1, 1, overrides={'F_MULT': 1,
                                             'Re': 0.1})),
       (set_ic, 'nl_6_HRx',
        build_interp_params(0.5, 1, overrides={'F_MULT': 1,
                                               'Re': 0.2})),
    ]
    if '-plot' in sys.argv:
        for _, name, params_dict in tasks:
            plot(name, params_dict)

    elif '-merge' in sys.argv:
        for _, name, _ in tasks:
            merge(name)

    elif '-write' in sys.argv:
        for _, name, params_dict in tasks:
            write_front(name, params_dict)

    elif '-front' in sys.argv:
        for _, name, params_dict in tasks:
            plot_front(name, params_dict)

    else:
        for task in tasks:
            run(*task)

# R = [0.084, 0.188, 0.265, 0.296]
# Re = [0.7, 0.4, 0.3, 0.2]
