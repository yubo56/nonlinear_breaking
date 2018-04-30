'''
2d_1_strat + Navier-Stokes dissipation
'''
from multiprocessing import Pool
import numpy as np
from strat_helper import *

N_PARALLEL = 20
H = 1
num_timesteps = 3e4

XMAX = H
ZMAX = 3 * H
KX = 4 * np.pi / H
KZ = -(np.pi) * np.pi * 2 / H
G = (KX**2 + KZ**2 + 1 / (4 * H**2)) / KX**2 * (2 * np.pi)**2 * H # omega = 2pi
OMEGA = get_omega(G, H, KX, KZ)
_, VPH_Z = get_vph(G, H, KX, KZ)
T_F = abs(ZMAX / VPH_Z) * 6
DT = T_F / num_timesteps
NUM_SNAPSHOTS = 400

PARAMS_RAW = {'XMAX': XMAX,
              'ZMAX': ZMAX,
              'N_X': 64,
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
              'SPONGE_STRENGTH': 6,
              'SPONGE_START': 0.7 * ZMAX,
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
        # linear, neumann works to drop ux waviness)
        (get_solver, sponge_lin, zero_ic,
         'sponge_lin',
         build_interp_params(4, 4)),

        # nonlin, low amplitude aite
        (get_solver, sponge_nonlin, zero_ic,
         'sponge_nonlin1',
         build_interp_params(4, 4)),

        # nonlin, how no disp = blowup
        (get_solver, sponge_nonlin, zero_ic,
         'sponge_nonlin2',
         build_interp_params(4, 4, overrides={'A': 0.02})),

        # linear, n-s works (has significant ux waves)
        (ns_get_solver, ns_sponge_lin, zero_ic,
         'ns_sponge_lin',
         build_interp_params(4, 4)),

        # linear, ramps up (smaller ux waves but still)
        (ns_get_solver, ns_sponge_lin_gradual, zero_ic,
         'ns_sponge_lin_gradual',
         build_interp_params(4, 4)),

        # nonlinear gradual, still blows up
        (ns_get_solver, ns_sponge_nonlin_gradual, zero_ic,
         'ns_sponge_nonlin_gradual',
         build_interp_params(4, 4, overrides={'A': 0.02})),

        # dz(uz_z) BC (force-like), still blows up
        (ns_get_solver, ns_sponge_nonlin2, zero_ic,
         'ns_sponge_nonlin1',
         build_interp_params(4, 4)),

        # use pressure to drive at boundary, linear
        (get_solver, sponge_lin_p_bc, zero_ic,
         'sponge_p_bc',
         build_interp_params(4, 4)),

        # use dz(pressure) to drive at boundary, nonlin
        (get_solver, sponge_nonlin_dp_bc, zero_ic,
         'sponge_p_bc_nonlin1',
         build_interp_params(4, 4)),

        # higher-A nonlin pressure drive
        (get_solver, sponge_nonlin_dp_bc, zero_ic,
         'sponge_p_bc_nonlin2',
         build_interp_params(4, 4, overrides={'A': 0.02})),

        # (rad_bc, zero_ic, 'rad', build_interp_params(8, 2)),
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
