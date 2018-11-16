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

NUM_SNAPSHOTS = 400
TARGET_DISP_RAT = 0.7

PARAMS_RAW = {'XMAX': XMAX,
              'ZMAX': ZMAX,
              'N_X': 64,
              'N_Z': 256,
              'KX': 2 * np.pi / XMAX,
              'KZ': -10 * np.pi / H,
              'H': H,
              'RHO0': 1,
              'Z0': 0.15 * ZMAX,
              'SPONGE_STRENGTH': 10,
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
    T_F = abs(ZMAX / VG_Z) * 20

    params['T_F'] = T_F
    params['g'] = g
    params['OMEGA'] = OMEGA
    params['S'] = params['ZMAX'] / 512 * 4
    params['INTERP_X'] = interp_x
    params['INTERP_Z'] = interp_z
    params['N_X'] //= interp_x
    params['N_Z'] //= interp_z
    params['DT'] = min(0.1 / OMEGA, 0.1)
    params['F'] = params.get('F_MULT', 0.1) * \
        (TARGET_DISP_RAT * OMEGA / KZ) / get_uz_f_ratio(params)
    # for nabla^n visc, u / (nu * kx^{n-1}) = 1
    params['NU'] = params.get('NU_MULT', 1) * \
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
        (set_ic, 'vstrat_low',
         build_interp_params(1, 1, overrides={'NU_MULT': 40,
                                              'USE_CFL': True,
                                              'UZ0_COEFF': 0.3})),
        (set_ic, 'vstrat',
         build_interp_params(1, 1, overrides={'NU_MULT': 40,
                                              'USE_CFL': True,
                                              'UZ0_COEFF': 1})),
    ]
    if '-plot' in sys.argv:
        for _, name, params_dict in tasks:
            plot(name, params_dict)

    elif '-merge' in sys.argv:
        for _, name, _ in tasks:
            merge(name)

    else:
        for task in tasks:
            run(*task)
