'''
2d_1_strat + Navier-Stokes dissipation
'''
from multiprocessing import Pool
import numpy as np
from strat_helper import *

N_PARALLEL = 20
H = 1
num_timesteps = 5e2

XMAX = H
ZMAX = 3 * H
KX = 2 * np.pi / H
KZ = -(np.pi) * np.pi / H
G = (KX**2 + KZ**2 + 1 / (4 * H**2)) / KX**2 * (2 * np.pi)**2 * H # omega = 2pi
OMEGA = get_omega(G, H, KX, KZ)
_, VPH_Z = get_vph(G, H, KX, KZ)
T_F = abs(ZMAX / VPH_Z) * 2
DT = T_F / num_timesteps
NUM_SNAPSHOTS = 100

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
              'F': 0.01,
              'SPONGE_STRENGTH': 30,
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
    params['NU'] = 0.1 * (ZMAX / params['N_Z'])**2 / np.pi**2 # smallest wavenumber
    return params

def run(get_solver, bc, ic, name, params_dict):
    assert 'INTERP_X' in params_dict and 'INTERP_Z' in params_dict,\
        'params need INTERP'
    run_strat_sim(get_solver, bc, ic, name, params_dict)
    return '%s completed' % name

if __name__ == '__main__':
    tasks = [
        # linear, neumann works to drop ux waviness
        (get_solver, setup_problem, zero_ic,
         'F1',
         build_interp_params(4, 4)),
        # (get_solver, setup_problem, zero_ic,
        #  'F2',
        #  build_interp_params(4, 4, overrides={'F': 0.3})),
        # (get_solver, setup_problem, zero_ic,
        #  'F3',
        #  build_interp_params(4, 4, overrides={'F': 0.6})),
        # (get_solver, setup_problem, zero_ic,
        #  'F4',
        #  build_interp_params(4, 4, overrides={'F': 1.0})),
        # (get_solver, setup_problem, zero_ic,
        #  'F5',
        #  build_interp_params(4, 4, overrides={'F': 2.0})),
        # (get_solver, setup_problem, zero_ic,
        #  'F6',
        #  build_interp_params(4, 4, overrides={'F': 4.0})),
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
