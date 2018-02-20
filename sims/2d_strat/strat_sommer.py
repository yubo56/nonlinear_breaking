'''
Almost the same as strat.py, just different BC

Incompressible fluid equations w/ vertical stratification
- div u1 = 0
- dt (rho1) - rho0 * u1z / H = 0
- dt(u1) = -grad(P1) / rho0 - rho1 * g / rho0

Periodic BC in x, radiative at z=L, Driving term at z=0
'''
import logging
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
from dedalus import public as de
from dedalus.extras.plot_tools import quad_mesh, pad_limits

XMAX = 10
ZMAX = 20
N_X = 64
N_Z = 256
DX = XMAX / N_X
DZ = ZMAX / N_Z

T_F = 200
DT = 2e-2
KX = 2
KZ = 1
H = 5
G = 10
N = G / H
RHO0 = 1

NUM_SNAPSHOTS = 800

if __name__ == '__main__':
    # Bases and domain
    x_basis = de.Fourier('x', N_X, interval=(0, XMAX), dealias=3/2)
    z_basis = de.Chebyshev('z', N_Z, interval=(0, ZMAX), dealias=3/2)
    domain = de.Domain([x_basis, z_basis], np.float64)

    problem = de.IVP(domain, variables=['P', 'rho', 'ux', 'uz'])
    problem.meta['uz']['z']['dirichlet'] = True
    problem.parameters['rho0'] = RHO0
    problem.parameters['L'] = XMAX
    problem.parameters['g'] = G
    problem.parameters['KX'] = KX
    problem.parameters['H'] = H
    problem.parameters['N'] = N
    problem.parameters['omega'] = np.sqrt(
        N**2 * KX**2 / (KX**2 + KZ**2 + 0.25 / H**2))
    problem.add_equation("dx(ux) + dz(uz) = 0")
    problem.add_equation("dt(rho) - rho0 * uz / H = 0")
    problem.add_equation("dt(ux) + dx(P)/rho0 = 0")
    problem.add_equation("dt(uz) + dz(P)/rho0 + rho * g / rho0= 0")
    problem.add_bc("right(dt(uz) * N * KX / omega ** 2 - dz(uz)) = 0",
                   condition="nx != 0")
    problem.add_bc("left(P) = 0", condition="nx == 0")
    problem.add_bc("left(uz) = sin(6.28 * x / L - omega * t)")

    # Build solver
    solver = problem.build_solver(de.timesteppers.RK222)
    solver.stop_sim_time = T_F
    solver.stop_wall_time = np.inf
    solver.stop_iteration = np.inf

    # Initial conditions
    x = domain.grid(0)
    z = domain.grid(1)
    ux = solver.state['ux']
    uz = solver.state['uz']
    P = solver.state['P']
    rho = solver.state['rho']

    # need to reverse for some reason?
    gshape = domain.dist.grid_layout.global_shape(scales=1)
    P['g'] = np.zeros(gshape)
    ux['g'] = np.zeros(gshape)
    uz['g'] = np.zeros(gshape)
    rho['g'] = np.zeros(gshape)

    snapshots = solver.evaluator.add_file_handler('snapshots',
                                                  sim_dt=T_F / NUM_SNAPSHOTS)
    snapshots.add_system(solver.state)

    # Main loop
    while solver.ok:
        solver.step(DT)
        curr_iter = solver.iteration

        # if curr_iter % int((T_F / DT) / NUM_SNAPSHOTS) == 0:
        #     ux.set_scales(1, keep_data=True)
        #     uz.set_scales(1, keep_data=True)
        #     P.set_scales(1, keep_data=True)
        #     rho.set_scales(1, keep_data=True)

        #     # plot vars
        #     xmesh, zmesh = quad_mesh(x=x[:, 0], y=z[0])
        #     for var, name in [(ux['g'], 'ux'), (uz['g'], 'uz'),
        #                       (rho['g'], 'rho'), (P['g'], 'P')]:
        #         f = plt.figure()
        #         plt.pcolormesh(xmesh, zmesh, np.transpose(var), cmap='YlGnBu')
        #         plt.axis(pad_limits(xmesh, zmesh))
        #         plt.colorbar()
        #         plt.xlabel('x')
        #         plt.ylabel('z')
        #         plt.title('%s, (t = %.2f)' % (name, curr_iter * DT))
        #         filename = 'plots/strat_sommer_%s_t%05.1f.png' % (name, curr_iter * DT)
        #         logger.info('Saving %s' % filename)
        #         plt.savefig(filename)
        #         plt.clf()
        #         plt.close(f)
