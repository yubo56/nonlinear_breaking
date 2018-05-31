'''
2d_1_strat + Navier-Stokes dissipation
'''
from multiprocessing import Pool
import numpy as np
from strat_helper import *

N_PARALLEL = 20
XMAX = 1
KX = 2 * np.pi / XMAX
ZMAX = 20 * (2 * np.pi / KX)
H = 10 / KX
KZ = -0.4 * KX

G = 1
OMEGA = get_omega(G, H, KX, KZ)
_, VPH_Z = get_vph(G, H, KX, KZ)

num_timesteps = 1e3
T_F = abs(ZMAX / VPH_Z)
DT = T_F / num_timesteps
NUM_SNAPSHOTS = 500

PARAMS_RAW = {'XMAX': XMAX,
              'ZMAX': ZMAX,
              'N_X': 128,
              'N_Z': 256,
              'T_F': T_F,
              'DT': DT,
              'OMEGA': OMEGA,
              'KX': KX,
              'KZ': KZ,
              'H': H,
              'RHO0': 1,
              'G': G,
              'A': 0.005,
              'F': 0.1,
              'SPONGE_STRENGTH': 50,
              'SPONGE_START_HIGH': 0.8 * ZMAX,
              'SPONGE_START_LOW': 0.2 * ZMAX,
              'NUM_SNAPSHOTS': NUM_SNAPSHOTS}

def build_interp_params(interp_x, interp_z, dt=DT, overrides=None):
    params = {**PARAMS_RAW, **(overrides or {})}
    params['INTERP_X'] = interp_x
    params['INTERP_Z'] = interp_z
    params['N_X'] //= interp_x
    params['N_Z'] //= interp_z
    params['DT'] = dt
    params['NU'] = 0.01 * (ZMAX / params['N_Z'])**2 / np.pi**2 # smallest wavenumber
    return params

def run(get_solver, bc, ic, name, params_dict):
    assert 'INTERP_X' in params_dict and 'INTERP_Z' in params_dict,\
        'params need INTERP'
    run_strat_sim(get_solver, bc, ic, name, params_dict)
    return '%s completed' % name

if __name__ == '__main__':
    tasks = [
        (get_solver, setup_problem_unforced, wavepacket_ic,
         'wavepacket',
         build_interp_params(8, 4)),
    ]
    if len(tasks) == 1:
        run(*tasks[0])
    else:
        with Pool(processes=N_PARALLEL) as p:
            res = []
            for task in tasks:
                res.append(p.apply_async(run, task))

            for r in res:
                print(r.get())

    for get_solver, bc, _, name, params_dict in tasks:
        plot(get_solver, bc, name, params_dict)
