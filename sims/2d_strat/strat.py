'''
Incompressible fluid equations w/ vertical stratification
- div u1 = 0
- dt (rho1) - rho0 * u1z / H = 0
- dt(u1) = -grad(P1) / rho0 - rho1 * g / rho0

Periodic BC in x, Dirichlet/Neumann 0 at z=L, Driving term at z=0
'''
import time

from multiprocessing import Pool
import numpy as np
import strat_helper

N_PARALLEL = 8
START_DELAY = 10 # sleep so h5py has time to claim snapshots
if __name__ == '__main__':
    H = 1
    num_timesteps = 1e4

    XMAX = H
    ZMAX = 5 * H
    KX = -2 * np.pi / H
    KZ = 2 * np.pi / H
    G = 10
    OMEGA = strat_helper.get_omega(G, H, KX, KZ)
    T_F = ZMAX / (OMEGA/KZ) * 5
    DT = T_F / 3e4

    params = {'XMAX': XMAX,
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
              'NUM_SNAPSHOTS': 200}
    # modified sponge params
    params_sponge = dict(params)
    params_sponge['DT'] = 10 * DT
    params['N_X'] = 32
    params['N_Z'] = 128

    def dirichlet_bc(problem, *_):
        strat_helper.default_problem(problem)
        problem.add_bc('right(uz) = 0', condition='nx != 0')

    def neumann_bc(problem, *_):
        strat_helper.default_problem(problem)
        problem.add_bc('right(dz(uz)) = 0', condition='nx != 0')

    def rad_bc(problem, *_):
        strat_helper.default_problem(problem)
        problem.add_bc('right(dt(uz) + omega / KZ * dz(uz)) = 0',
                       condition='nx != 0')

    def sponge(problem, domain):
        '''
        puts a -gamma(z) * q damping on all dynamical variables, where gamma(z)
        is the sigmoid: damping * exp(steep * (z - z_sigmoid)) / (1 + exp(...))
        '''
        zmax = params['ZMAX']
        damp_start = zmax * 0.7 # start damping zone
        z = domain.grid(1)

        # sponge field
        sponge = domain.new_field()
        sponge.meta['x']['constant'] = True
        sponge['g'] = np.minimum(
            1 - (z - zmax)**2 / (damp_start - zmax)**2,
            np.zeros(np.shape(z)))

        problem.parameters['sponge'] = sponge
        problem.add_equation("dx(ux) + dz(uz) = 0")
        problem.add_equation("dt(rho) - rho0 * uz / H - sponge * rho= 0")
        problem.add_equation("dt(ux) + dx(P) / rho0 - sponge * ux= 0")
        problem.add_equation(
            "dt(uz) + dz(P) / rho0 + rho * g / rho0 - sponge * uz= 0")

        problem.add_bc("left(P) = 0", condition="nx == 0")
        problem.add_bc("left(uz) = cos(KX * x - omega * t)")
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

    def steady_ic(solver, domain):
        ux = solver.state['ux']
        uz = solver.state['uz']
        P = solver.state['P']
        rho = solver.state['rho']
        x = domain.grid(0)
        z = domain.grid(1)

        rho0 = params['RHO0']

        common_factor = np.exp(z / (2 * H)) * np.sin(KZ * (ZMAX - z)) / \
            np.sin(KZ * ZMAX)
        uz['g'] = np.cos(KX * x) * common_factor
        ux['g'] = -KZ / KX * np.cos(KX * x + 1 / (2 * H * KZ)) * common_factor
        rho['g'] = -rho0 / (H * OMEGA) * np.sin(KX * x) * common_factor
        P['g'] = rho0 * OMEGA * KZ / KX**2 *np.cos(KX * x + 1 / (2 * H * KZ)) \
            * common_factor

    def run(bc, ic, name, params_dict):
        strat_helper.run_strat_sim(bc, ic, name=name, **params_dict)
        return '%s completed' % name

    with Pool(processes=N_PARALLEL) as p:
        tasks = [
            (dirichlet_bc, zero_ic, 'd0', params), # strat_dirichlet.mp4
            (neumann_bc, zero_ic, 'n0', params), # strat_neumann.mp4
            (dirichlet_bc, steady_ic, 'dss', params), # strat_dirichlet_ss.mp4
            (sponge, zero_ic, 'sponge', params_sponge), # strat_sponge.mp4
        ]
        res = []
        for task in tasks:
            res.append(p.apply_async(run, task))
            time.sleep(START_DELAY)

        for r in res:
            print(r.get())
