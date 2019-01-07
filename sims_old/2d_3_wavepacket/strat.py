'''
2d_1_strat + Navier-Stokes dissipation
'''
from multiprocessing import Pool
import numpy as np
from strat_helper import *

N_PARALLEL = 20
XMAX = 1
KX = 2 * np.pi / XMAX
ZMAX = 30 * (2 * np.pi / KX)
H = 10 / KX
KZ = -0.4 * KX

G = 1
OMEGA = get_omega(G, H, KX, KZ)
_, VPH_Z = get_vph(G, H, KX, KZ)

num_timesteps = 1e3
T_F = abs(ZMAX / VPH_Z) * 0.4
DT = T_F / num_timesteps
NUM_SNAPSHOTS = 200

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
              'SPONGE_STRENGTH': 100,
              'SPONGE_START_HIGH': 0.9 * ZMAX,
              'SPONGE_START_LOW': 0.1 * ZMAX,
              'NUM_SNAPSHOTS': NUM_SNAPSHOTS}

def build_interp_params(interp_x, interp_z, dt=DT, overrides=None):
    params = {**PARAMS_RAW, **(overrides or {})}
    params['INTERP_X'] = interp_x
    params['INTERP_Z'] = interp_z
    params['N_X'] //= interp_x
    params['N_Z'] //= interp_z
    params['DT'] = dt
    params['NU'] = (ZMAX / params['N_Z'])**2 / np.pi**2 # ~ 1/k_max
    return params

def run(get_solver, bc, ic, name, params_dict):
    assert 'INTERP_X' in params_dict and 'INTERP_Z' in params_dict,\
        'params need INTERP'
    run_strat_sim(get_solver, bc, ic, name, params_dict)
    return '%s completed' % name

if __name__ == '__main__':
    tasks = [
        (get_solver, setup_problem_unforced, wavepacket_ic,
         'SBDF1',
         build_interp_params(4, 1, overrides={'TIMESTEPPER':
                                              de.timesteppers.SBDF1})),
        (get_solver, setup_problem_unforced, wavepacket_ic,
         'CNAB2',
         build_interp_params(4, 1, overrides={'TIMESTEPPER':
                                              de.timesteppers.CNAB2})),
        (get_solver, setup_problem_unforced, wavepacket_ic,
         'MCNAB2',
         build_interp_params(4, 1, overrides={'TIMESTEPPER':
                                              de.timesteppers.MCNAB2})),
        (get_solver, setup_problem_unforced, wavepacket_ic,
         'SBDF2',
         build_interp_params(4, 1, overrides={'TIMESTEPPER':
                                              de.timesteppers.SBDF2})),
        (get_solver, setup_problem_unforced, wavepacket_ic,
         'CNLF2',
         build_interp_params(4, 1, overrides={'TIMESTEPPER':
                                              de.timesteppers.CNLF2})),
        (get_solver, setup_problem_unforced, wavepacket_ic,
         'SBDF3',
         build_interp_params(4, 1, overrides={'TIMESTEPPER':
                                              de.timesteppers.SBDF3})),
        (get_solver, setup_problem_unforced, wavepacket_ic,
         'SBDF4',
         build_interp_params(4, 1, overrides={'TIMESTEPPER':
                                              de.timesteppers.SBDF4})),
        (get_solver, setup_problem_unforced, wavepacket_ic,
         'RK111',
         build_interp_params(4, 1, overrides={'TIMESTEPPER':
                                              de.timesteppers.RK111})),
        (get_solver, setup_problem_unforced, wavepacket_ic,
         'RK222',
         build_interp_params(4, 1, overrides={'TIMESTEPPER':
                                              de.timesteppers.RK222})),
        (get_solver, setup_problem_unforced, wavepacket_ic,
         'RK443',
         build_interp_params(4, 1, overrides={'TIMESTEPPER':
                                              de.timesteppers.RK443})),
        (get_solver, setup_problem_unforced, wavepacket_ic,
         'RKSMR',
         build_interp_params(4, 1, overrides={'TIMESTEPPER':
                                              de.timesteppers.RKSMR})),
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
