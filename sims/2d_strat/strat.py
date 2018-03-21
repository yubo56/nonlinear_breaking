'''
Incompressible fluid equations w/ vertical stratification
- div u1 = 0
- dt (rho1) - rho0 * u1z / H = 0
- dt(u1) = -grad(P1) / rho0 - rho1 * g / rho0

Periodic BC in x, Dirichlet/Neumann 0 at z=L, Driving term at z=0
creates h5 snapshot, then plots. if snapshot exists, skips computation
'''
from multiprocessing import Pool
import numpy as np
import strat_helper

N_PARALLEL = 8
START_DELAY = 10 # sleep so h5py has time to claim snapshots
H = 1
num_timesteps = 4e4

XMAX = H
ZMAX = 5 * H
KX = -2 * np.pi / H
KZ = (np.pi / 2) * np.pi / H
G = 10
OMEGA = strat_helper.get_omega(G, H, KX, KZ)
VPH_X, VPH_Z = strat_helper.get_vph(G, H, KX, KZ)
T_F = (ZMAX / VPH_Z) * 12
DT = T_F / num_timesteps

PARAMS_RAW = {'XMAX': XMAX,
              'ZMAX': ZMAX,
              'N_X': 64,
              'N_Z': 256,
              'T_F': T_F,
              'DT': DT,
              'KX': KX,
              'KZ': KZ,
              'H': H,
              'RHO0': 1,
              'G': G,
              'A': 0.05,
              'NUM_SNAPSHOTS': 200}

def build_interp_params(interp_x, interp_z, dt=DT):
    params = dict(PARAMS_RAW)
    params['INTERP_X'] = interp_x
    params['INTERP_Z'] = interp_z
    params['N_X'] //= interp_x
    params['N_Z'] //= interp_z
    params['DT'] = dt
    return params

def dirichlet_bc(problem, *_):
    strat_helper.default_problem(problem)
    problem.add_bc('right(uz) = 0', condition='nx != 0')

def rad_bc(problem, *_):
    # TODO incorrect
    strat_helper.default_problem(problem)
    problem.add_bc('right(dt(uz) + omega / KZ * dz(uz)) = 0',
                   condition='nx != 0')

def sponge(problem, domain):
    '''
    puts a -gamma(z) * q damping on all dynamical variables, where gamma(z)
    is the sigmoid: damping * exp(steep * (z - z_sigmoid)) / (1 + exp(...))
    '''
    zmax = PARAMS_RAW['ZMAX']
    damp_start = zmax * 0.7 # start damping zone
    z = domain.grid(1)

    # sponge field
    sponge = domain.new_field()
    sponge.meta['x']['constant'] = True
    sponge['g'] = np.maximum(
        1 - (z - zmax)**2 / (damp_start - zmax)**2,
        np.zeros(np.shape(z)))

    problem.parameters['sponge'] = sponge
    problem.add_equation("dx(ux) + dz(uz) = 0")
    problem.add_equation("dt(rho) - rho0 * uz / H + sponge * rho= 0")
    problem.add_equation("dt(ux) + dx(P) / rho0 + sponge * ux= 0")
    problem.add_equation(
        "dt(uz) + dz(P) / rho0 + rho * g / rho0 + sponge * uz= 0")

    problem.add_bc("left(P) = 0", condition="nx == 0")
    problem.add_bc("left(uz) = A * cos(KX * x - omega * t)")
    problem.add_bc('right(uz) = 0', condition='nx != 0')

def zero_ic(solver, domain):
    ux = solver.state['ux']
    uz = solver.state['uz']
    P = solver.state['P']
    rho = solver.state['rho']
    gshape = domain.dist.grid_layout.global_shape(scales=1)

    P['g'] = np.zeros(gshape)
    ux['g'] = np.zeros(gshape)
    uz['g'] = np.zeros(gshape)
    rho['g'] = np.zeros(gshape)

def run(bc, ic, name, params_dict):
    assert 'INTERP_X' in params_dict and 'INTERP_Z' in params_dict,\
        'params need INTERP'
    strat_helper.run_strat_sim(bc, ic, name=name, **params_dict)

if __name__ == '__main__':
    tasks = [
        (dirichlet_bc, zero_ic, 'd0', build_interp_params(8, 2)),
        (sponge, zero_ic, 'sponge2', build_interp_params(8, 4)),
    ]

    with Pool(processes=N_PARALLEL) as p:
        res = []
        for task in tasks:
            res.append(p.apply_async(run, task))

        for r in res:
            print(r.get())

    for bc, _, name, params_dict in tasks:
        strat_helper.plot(bc, name=name, **params_dict)
