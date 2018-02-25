'''
Incompressible fluid equations w/ vertical stratification
- div u1 = 0
- dt (rho1) - rho0 * u1z / H = 0
- dt(u1) = -grad(P1) / rho0 - rho1 * g / rho0

Periodic BC in x, Dirichlet/Neumann 0 at z=L, Driving term at z=0
'''
import logging
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
from dedalus import public as de
from dedalus.extras.plot_tools import quad_mesh
from dedalus.extras.flow_tools import CFL

XMAX = 10
ZMAX = 20
N_X = 64
N_Z = 256
DX = XMAX / N_X
DZ = ZMAX / N_Z

T_F = 80
DT = 2e-2
KX = 2
KZ = 1
H = ZMAX / 3
G = 10
N = G / H
RHO0 = 1

NUM_SNAPSHOTS = 80

if __name__ == '__main__':
    # Bases and domain
    x_basis = de.Fourier('x', N_X, interval=(0, XMAX), dealias=3/2)
    z_basis = de.Chebyshev('z', N_Z, interval=(0, ZMAX), dealias=3/2)
    domain = de.Domain([x_basis, z_basis], np.float64)
    x = domain.grid(0)
    z = domain.grid(1)

    problem = de.IVP(domain, variables=['P', 'rho', 'ux', 'uz'])
    problem.meta['uz']['z']['dirichlet'] = True
    problem.parameters['L'] = XMAX
    problem.parameters['g'] = G
    problem.parameters['H'] = H
    problem.parameters['omega'] = np.sqrt(
        N**2 * KX**2 / (KX**2 + KZ**2 + 0.25 / H**2))

    # rho0 stratification
    rho0 = domain.new_field()
    rho0.meta['x']['constant'] = True
    rho0['g'] = RHO0 * np.exp(-z / H)
    xmesh, zmesh = quad_mesh(x=x[:,0], y=z[0])
    problem.parameters['rho0'] = rho0

    plt.pcolormesh(xmesh, zmesh, np.transpose(rho0['g']))
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title('Background rho0')
    plt.colorbar()
    plt.savefig('strat_rho0.png')

    problem.add_equation("dx(ux) + dz(uz) = 0")
    problem.add_equation("dt(rho) - rho0 * uz / H = 0")
    problem.add_equation("dt(ux) + dx(P) / rho0 = 0")
    problem.add_equation("dt(uz) + dz(P) / rho0 + rho * g / rho0 = 0")
    # problem.add_bc("right(uz) = 0", condition="nx != 0")
    problem.add_bc("right(dz(uz)) = 0", condition="nx != 0")
    problem.add_bc("left(P) = 0", condition="nx == 0")
    problem.add_bc("left(uz) = sin(6.28 * x / L - omega * t)")

    # Build solver
    solver = problem.build_solver(de.timesteppers.RK222)
    solver.stop_sim_time = T_F
    solver.stop_wall_time = np.inf
    solver.stop_iteration = np.inf

    # Initial conditions
    ux = solver.state['ux']
    uz = solver.state['uz']
    P = solver.state['P']
    rho = solver.state['rho']

    # use CFL
    cfl = CFL(solver, initial_dt=DT, cadence=10, max_dt=10 * DT, threshold=0.05)
    cfl.add_velocities(('ux', 'uz'))

    # slices so can mpirun
    gshape = domain.dist.grid_layout.global_shape(scales=1)
    slices = domain.dist.grid_layout.slices(scales=1)
    P['g'] = np.zeros(gshape)[slices]
    ux['g'] = np.zeros(gshape)[slices]
    uz['g'] = np.zeros(gshape)[slices]
    rho['g'] = np.zeros(gshape)[slices]

    snapshots = solver.evaluator.add_file_handler('snapshots',
                                                  sim_dt=T_F / NUM_SNAPSHOTS)
    snapshots.add_system(solver.state)

    # Main loop
    timesteps = []
    while solver.ok:
        # cfl_dt = cfl.compute_dt()
        cfl_dt = DT
        timesteps.append(cfl_dt)
        solver.step(cfl_dt)
        curr_iter = solver.iteration

        if curr_iter % int((T_F / DT) / NUM_SNAPSHOTS) == 0:
            logger.info('Average timestep at time %.2f: %f' %
                        (solver.sim_time, np.mean(timesteps)))
            timesteps = []
