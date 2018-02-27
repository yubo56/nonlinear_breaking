#!/usr/bin/env python
'''
helper function to run the shared stratification scenario. user just has to
specify BCs and ICs
'''
import logging
import os
import matplotlib.pyplot as plt
import numpy as np

from dedalus import public as de
from dedalus.extras.plot_tools import quad_mesh
# from dedalus.extras.flow_tools import CFL

SNAPSHOTS_DIR = 'snapshots'

def run_strat_sim(setup_problem,
                  set_ICs,
                  XMAX,
                  ZMAX,
                  N_X,
                  N_Z,
                  T_F,
                  DT,
                  KX,
                  KZ,
                  H_FACT,
                  RHO0,
                  NUM_SNAPSHOTS,
                  G,
                  name=None):
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
    H = ZMAX / H_FACT
    logger = logging.getLogger(name or __name__)

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
    problem.parameters['KX'] = KX
    problem.parameters['KZ'] = KZ
    problem.parameters['omega'] = np.sqrt(
        (G / H) * KX**2 / (KX**2 + KZ**2 + 0.25 / H**2))

    # rho0 stratification
    rho0 = domain.new_field()
    rho0.meta['x']['constant'] = True
    rho0['g'] = RHO0 * np.exp(-z / H)
    problem.parameters['rho0'] = rho0

    xmesh, zmesh = quad_mesh(x=x[:, 0], y=z[0])
    plt.pcolormesh(xmesh, zmesh, np.transpose(rho0['g']))
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title('Background rho0')
    plt.colorbar()
    plt.savefig('strat_rho0.png')

    setup_problem(problem, domain)

    # Build solver
    solver = problem.build_solver(de.timesteppers.RK222)
    solver.stop_sim_time = T_F
    solver.stop_wall_time = np.inf
    solver.stop_iteration = np.inf

    # Initial conditions
    set_ICs(solver, domain)

    # use CFL
    # cfl = CFL(solver, initial_dt=DT, cadence=10, max_dt=10 * DT, threshold=0.05)
    # cfl.add_velocities(('ux', 'uz'))

    snapshots = solver.evaluator.add_file_handler(SNAPSHOTS_DIR,
                                                  sim_dt=T_F / NUM_SNAPSHOTS)
    snapshots.add_system(solver.state)

    # Main loop
    timesteps = []
    logger.info('Starting sim...')
    while solver.ok:
        # cfl_dt = cfl.compute_dt()
        cfl_dt = DT
        timesteps.append(cfl_dt)
        solver.step(cfl_dt)
        curr_iter = solver.iteration

        if curr_iter % int((T_F / DT) / NUM_SNAPSHOTS) == 0:
            logger.info('Average timestep at time %.2f: %f',
                        solver.sim_time,
                        np.mean(timesteps))
            timesteps = []

def default_problem(problem):
    problem.add_equation("dx(ux) + dz(uz) = 0")
    problem.add_equation("dt(rho) - rho0 * uz / H = 0")
    problem.add_equation("dt(ux) + dx(P) / rho0 = 0")
    problem.add_equation("dt(uz) + dz(P) / rho0 + rho * g / rho0 = 0")
    problem.add_bc("left(P) = 0", condition="nx == 0")
    problem.add_bc("left(uz) = cos(KX * x - omega * t)")
