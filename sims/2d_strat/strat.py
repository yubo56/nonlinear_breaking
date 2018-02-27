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

N_PARALLEL = 8 # python refuses to kick off more than 2 here on my local...
START_DELAY = 10 # sleep so h5py has time to claim snapshots
if __name__ == '__main__':
    params = {'XMAX': 10,
              'ZMAX': 20,
              'N_X': 64,
              'N_Z': 256,
              'T_F': 120,
              'DT': 2e-2,
              'KX': 2 * np.pi / 10, # kx = 2pi/L_x
              'KZ': 1,
              'H_FACT': 2,
              'RHO0': 1,
              'G': 10,
              'NUM_SNAPSHOTS': 120}

    def dirichlet_bc(problem):
        strat_helper.default_problem(problem)
        problem.add_bc("right(uz) = 0", condition="nx != 0")

    def neumann_bc(problem):
        strat_helper.default_problem(problem)
        problem.add_bc("right(dz(uz)) = 0", condition="nx != 0")

    def rad_bc(problem):
        strat_helper.default_problem(problem)
        problem.add_bc("right(dt(uz) = omega / kz * dz(uz)")

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

        ZMAX = params['ZMAX']
        H = params['ZMAX'] / params['H_FACT']
        kx = params['KX']
        kz = params['KZ']
        rho0 = params['RHO0']
        g = params['G']
        omega = np.sqrt(g / H * kx**2 / (kx**2 + kz**2 + 1/(4 * H**2)))

        common_factor = np.exp(z / (2 * H)) * np.sin(kz * (ZMAX - z)) / \
            np.sin(kz * ZMAX)
        uz['g'] = np.cos(kx * x) * common_factor
        ux['g'] = -kz / kx * np.cos(kx * x + 1 / (2 * H * kz)) * common_factor
        rho['g'] = -rho0 / (H * omega) * np.sin(kx * x) * common_factor
        P['g'] = rho0 * omega * kz / kx**2 *np.cos(kx * x + 1 / (2 * H * kz)) \
            * common_factor

    def run(bc, ic, name):
        strat_helper.run_strat_sim(bc, ic, name=name, **params)
        return '%s completed' % name

    with Pool(processes=N_PARALLEL) as p:
        tasks = [
            # (dirichlet_bc, zero_ic, 'd0'), # strat_dirichlet_s1.mp4
            # (neumann_bc, zero_ic, 'n0'), # strat_neumann_s2.mp4
            # (dirichlet_bc, steady_ic, 'dss'), # strat_dirichlet_ss_s3.mp4
            (rad_bc, zero_ic, 'rad0'), # strat_rad_s4.mp4
        ]
        res = []
        for task in tasks:
            res.append(p.apply_async(run, task))
            time.sleep(START_DELAY)

        for r in res:
            print(r.get())
