'''
2d_1_strat + Navier-Stokes dissipation
'''
from multiprocessing import Pool
import numpy as np
import strat_helper

N_PARALLEL = 8
H = 1
num_timesteps = 3e4

XMAX = H
ZMAX = 2 * H
KX = 2 * np.pi / H
KZ = -(np.pi / 2) * np.pi / H
G = (KX**2 + KZ**2 + 1 / (4 * H**2)) / KX**2 * (2 * np.pi)**2 * H # omega = 2pi
OMEGA = strat_helper.get_omega(G, H, KX, KZ)
_, VPH_Z = strat_helper.get_vph(G, H, KX, KZ)
T_F = abs(ZMAX / VPH_Z) * 4
DT = T_F / num_timesteps

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
              'NUM_SNAPSHOTS': 400}

def build_interp_params(interp_x, interp_z, dt=DT, overrides=None):
    params = {**PARAMS_RAW, **(overrides or {})}
    params['INTERP_X'] = interp_x
    params['INTERP_Z'] = interp_z
    params['N_X'] //= interp_x
    params['N_Z'] //= interp_z
    params['DT'] = dt
    params['NU'] = 0.1 * (ZMAX / params['N_Z'])**2 / np.pi**2 # smallest wavenumber
    return params

def run(bc, ic, name, params_dict):
    assert 'INTERP_X' in params_dict and 'INTERP_Z' in params_dict,\
        'params need INTERP'
    strat_helper.run_strat_sim(bc, ic, name, params_dict)
    return '%s completed' % name

if __name__ == '__main__':
    tasks = [
        (strat_helper.sponge_lin,
         strat_helper.zero_ic,
         'sponge_lin',
         build_interp_params(8, 4)),
        (strat_helper.sponge_nonlin,
         strat_helper.bg_ic,
         'sponge_nonlin1',
         build_interp_params(8, 4)),
        (strat_helper.sponge_nonlin,
         strat_helper.bg_ic,
         'sponge_nonlin2',
         build_interp_params(8, 4, {'A': 0.05})),
        # (rad_bc, zero_ic, 'rad', build_interp_params(8, 4)),
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

    for bc, _, name, params_dict in tasks:
        strat_helper.plot(bc, name, params_dict)
