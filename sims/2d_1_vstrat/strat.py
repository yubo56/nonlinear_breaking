'''
for i in {1..4}; do rm -f *.png && rsync exo15c:~/research/nonlinear_breaking/sims/2d_1_vstrat/snapshots_vstrat$i/*.png . && ffmpeg -y -framerate 12 -i 't_%d.png' vstrat$i.mp4; done; rm -f *.png

'''
import numpy as np
import sys
from strat_helper import *
from mpi4py import MPI
CW = MPI.COMM_WORLD

H = 1
XMAX = H
ZMAX = H

NUM_SNAPSHOTS = 300

PARAMS_DEFAULT = {'XMAX': XMAX,
                  'ZMAX': ZMAX,
                  'N_X': 128,
                  'N_Z': 256,

              'KX': 2 * np.pi / XMAX,
              'KZ': -2 * np.pi / (H / 8),
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
                  'T_MULT': 5,

                  'NUM_SNAPSHOTS': NUM_SNAPSHOTS,
                  'NL': True}

def get_params(overrides=None):
    params = {**PARAMS_DEFAULT, **(overrides or {})}
    g = params['g']
    KX = params['KX']
    KZ = params['KZ']
    OMEGA = get_omega(g, H, KX, KZ)
    VG_Z = get_vgz(g, H, KX, KZ)

    PARAMS_DEFAULT['T_F'] = abs(ZMAX / VG_Z) * params['T_MULT']
    PARAMS_DEFAULT['OMEGA'] = OMEGA
    PARAMS_DEFAULT['S'] = abs(1 / params['KZ'])
    PARAMS_DEFAULT['DT'] = min(0.1 / OMEGA, 0.1)

    # second override unfortunately
    params = {**PARAMS_DEFAULT, **(overrides or {})}

    params['F'] = params['F_MULT'] * \
        (OMEGA / KZ) / get_uz_f_ratio(params) \
    # for nabla^n visc, u / (nu * kx^{n-1}) = 1
    params['NU'] = params['Re_inv'] * \
        OMEGA * (params['ZMAX'] / (2 * np.pi * params['N_Z']))**5 / abs(KZ)

    if CW.rank == 0: # print only on root process
        print(params)
    return params

if __name__ == '__main__':
    tasks = [
        ('vstrat_lin',
         get_params(overrides={'Re_inv': 200,
                               'F_MULT': 0.01,
                               'T_MULT': 4,
                               'NL': True,
                               'N_X': 64,
                               'UZ0_COEFF': 0})),
        ('vstrat_nl_4',
         get_params(overrides={'Re_inv': 300,
                               'F_MULT': 0.02,
                               'T_MULT': 4,
                               'NL': True,
                               'UZ0_COEFF': 0.7})),
        ('vstrat_nl_2',
         get_params(overrides={'Re_inv': 6e3,
                               'F_MULT': 0.08,
                               'T_MULT': 4,
                               'NL': True,
                               'UZ0_COEFF': 1.1})),
        ('vstrat_nl_2_highA',
         get_params(overrides={'Re_inv': 6e3,
                               'F_MULT': 0.15,
                               'T_MULT': 4,
                               'NL': True,
                               'UZ0_COEFF': 1.1})),
        ('vstrat_nl_2_highRe',
         get_params(overrides={'Re_inv': 1e3,
                               'F_MULT': 0.08,
                               'T_MULT': 4,
                               'NL': True,
                               'UZ0_COEFF': 1.1})),
        ('vstrat_nl_2_highA_Re',
         get_params(overrides={'Re_inv': 1e3,
                               'F_MULT': 0.15,
                               'T_MULT': 4,
                               'NL': True,
                               'UZ0_COEFF': 1.1})),
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
