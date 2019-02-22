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

PARAMS_RAW = {'XMAX': XMAX,
              'ZMAX': ZMAX,
              'N_X': 256,
              'N_Z': 1024,
              'KX': 2 * np.pi / XMAX,
              'KZ': -2 * np.pi / (H / 8),
              'H': H,
              'RHO0': 1,
              'Z0': 0.2 * ZMAX,
              'SPONGE_STRENGTH': 5,
              'SPONGE_WIDTH': 0.6,
              'SPONGE_HIGH': 0.95 * ZMAX,
              'SPONGE_LOW': 0.05 * ZMAX,
              'NUM_SNAPSHOTS': NUM_SNAPSHOTS}

def build_interp_params(interp_x, interp_z, overrides=None):
    params = {**PARAMS_RAW, **(overrides or {})}
    KX = params['KX']
    KZ = params['KZ']
    g = H # N^2 = 1

    OMEGA = get_omega(g, H, KX, KZ)
    VG_Z = get_vgz(g, H, KX, KZ)
    T_F = abs(ZMAX / VG_Z) * params.get('T_MULT', 5)

    params['T_F'] = T_F
    params['g'] = g
    params['OMEGA'] = OMEGA
    params['S'] = 1 / params['KZ'] # params['ZMAX'] / params['N_Z'] * 2
    params['INTERP_X'] = interp_x
    params['INTERP_Z'] = interp_z
    # omega * DT << 1 is required, as is DT << 1/N = 1
    params['DT'] = params.get('DT', min(0.1 / OMEGA, 0.1))
    params['F'] = params.get('F_MULT', 1) * \
        (OMEGA / KZ) / get_uz_f_ratio(params) \
    # for nabla^n visc, u / (nu * kx^{n-1}) = 1
    params['NU'] = params.get('Re', 1) * \
        OMEGA * (params['ZMAX'] / (2 * np.pi * params['N_Z']))**5 / abs(KZ)

    params['UZ0_COEFF'] = params.get('UZ0_COEFF', 1)
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
        (set_ic, 'vstrat_lin',
         build_interp_params(4, 4, overrides={'Re': 200,
                                              'F_MULT': 0.01,
                                              'T_MULT': 3,
                                              'NL': True,
                                              'UZ0_COEFF': 0})),
        (set_ic, 'vstrat',
         build_interp_params(4, 4, overrides={'Re': 200,
                                              'F_MULT': 0.05,
                                              'T_MULT': 10,
                                              'NL': True,
                                              'UZ0_COEFF': 1.1})),
    ]
    if '-plot' in sys.argv:
        for _, name, params_dict in tasks:
            plot(name, params_dict)

    elif '-merge' in sys.argv:
        for _, name, _ in tasks:
            merge(name)

    elif '-front' in sys.argv:
        for _, name, params_dict in tasks:
            plot_front(name, params_dict)

    else:
        for task in tasks:
            run(*task)
