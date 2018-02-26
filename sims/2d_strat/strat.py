'''
Incompressible fluid equations w/ vertical stratification
- div u1 = 0
- dt (rho1) - rho0 * u1z / H = 0
- dt(u1) = -grad(P1) / rho0 - rho1 * g / rho0

Periodic BC in x, Dirichlet/Neumann 0 at z=L, Driving term at z=0
'''
import numpy as np
from strat_helper import run_strat_sim

if __name__ == '__main__':
    params = {'H_FACT': 3,
              'ZMAX': 20,
              'XMAX': 10,
              'KX': 2 * np.pi / 10, # 2pi/L_X
              'KZ': 1,
              'RHO0': 1,
              'G': 10}

    def dirichlet_bc(problem):
        problem.add_bc("right(uz) = 0", condition="nx != 0")

    def neumann_bc(problem):
        problem.add_bc("right(dz(uz)) = 0", condition="nx != 0")

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

    def steady_dirichlet_ic(solver, domain):
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
        omega = g / H * kx**2 / (kx**2 + kz**2 + 1/(4 * H**2))

        common_factor = np.exp(z / (2 * H)) * np.sin(kz * (ZMAX - z)) / \
            np.sin(kz * ZMAX)
        uz['g'] = common_factor * np.cos(kx * x)
        ux['g'] = -kz / kx * np.cos(kx * x + 1 / (2 * H * kz))
        rho['g'] = -rho0 / (H * omega) * np.sin(kx * x)
        P['g'] = rho0 * omega * kz / kx**2 *np.cos(kx * x + 1 / (2 * H * kz))

    # strat_dirichlet_s1.mp4
    # run_strat_sim(dirichlet_bc, zero_ic, **params)

    # strat_neumann_s2.mp4
    # run_strat_sim(neumann_bc, zero_ic, **params)

    # strat_dirichlet_ss_s3.mp4
    run_strat_sim(neumann_bc, steady_dirichlet_ic, **params)
