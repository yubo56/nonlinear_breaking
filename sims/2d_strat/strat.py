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
        steep = 5 # steepness of sigmoid transition
        z_sigmoid = params['ZMAX'] * 0.7 # location of sigmoid transition
        damping = 3

        z = domain.grid(1)

        # sponge field
        sponge = domain.new_field()
        sponge.meta['x']['constant'] = True
        sig_exp = np.exp(steep * (z - z_sigmoid))
        sponge['g'] = damping * sig_exp / (1 + sig_exp)

        problem.parameters['sponge'] = sponge
        problem.add_equation("dx(ux) + dz(uz) = 0")
        problem.add_equation("dt(rho) - rho0 * uz / H + sponge * rho= 0")
        problem.add_equation("dt(ux) + dx(P) / rho0 + sponge * ux= 0")
        problem.add_equation(
            "dt(uz) + dz(P) / rho0 + rho * g / rho0 + sponge * uz= 0")

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
            # (dirichlet_bc, zero_ic, 'd0'), # strat_dirichlet.mp4
            # (neumann_bc, zero_ic, 'n0'), # strat_neumann.mp4
            # (dirichlet_bc, steady_ic, 'dss'), # strat_dirichlet_ss.mp4
            # (rad_bc, zero_ic, 'rad'), # strat_rad.mp4
            (sponge, zero_ic, 'sponge'), # strat_sponge.mp4
        ]
        res = []
        for task in tasks:
            res.append(p.apply_async(run, task))
            time.sleep(START_DELAY)

        for r in res:
            print(r.get())
