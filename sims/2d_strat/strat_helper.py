#!/usr/bin/env python
'''
helper function to run the shared stratification scenario. user just has to
specify BCs and ICs
'''
import h5py
import logging
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict
from dedalus import public as de
from dedalus.extras.plot_tools import quad_mesh, pad_limits

SNAPSHOTS_DIR = 'snapshots_%s'
def get_solver(setup_problem,
               XMAX,
               ZMAX,
               N_X,
               N_Z,
               T_F,
               KX,
               KZ,
               H,
               RHO0,
               G):
    # Bases and domain
    x_basis = de.Fourier('x', N_X, interval=(0, XMAX), dealias=3/2)
    z_basis = de.Chebyshev('z', N_Z, interval=(0, ZMAX), dealias=3/2)
    domain = de.Domain([x_basis, z_basis], np.float64)
    z = domain.grid(1)

    problem = de.IVP(domain, variables=['P', 'rho', 'ux', 'uz'])
    problem.meta['uz']['z']['dirichlet'] = True
    problem.parameters['L'] = XMAX
    problem.parameters['g'] = G
    problem.parameters['H'] = H
    problem.parameters['KX'] = KX
    problem.parameters['KZ'] = KZ
    problem.parameters['omega'] = get_omega(G, H, KX, KZ)

    # rho0 stratification
    rho0 = domain.new_field()
    rho0.meta['x']['constant'] = True
    rho0['g'] = RHO0 * np.exp(-z / H)
    problem.parameters['rho0'] = rho0

    # plot rho0
    # xmesh, zmesh = quad_mesh(x=x[:, 0], y=z[0])
    # plt.pcolormesh(xmesh, zmesh, rho0['g'].T)
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.title('Background rho0')
    # plt.colorbar()
    # plt.savefig('strat_rho0.png')

    setup_problem(problem, domain)

    # Build solver
    solver = problem.build_solver(de.timesteppers.RK222)
    solver.stop_sim_time = T_F
    solver.stop_wall_time = np.inf
    solver.stop_iteration = np.inf
    return solver, domain

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
                  H,
                  RHO0,
                  NUM_SNAPSHOTS,
                  G,
                  name=None):
    snapshots_dir = SNAPSHOTS_DIR % name
    try:
        os.makedirs(snapshots_dir)
    except FileExistsError:
        print('snapshots already exist, exiting...')
        return
    logger = logging.getLogger(name or __name__)

    solver, domain = get_solver(
        setup_problem,
        XMAX, ZMAX, N_X, N_Z, T_F, KX, KZ, H, RHO0, G)

    # Initial conditions
    set_ICs(solver, domain)

    snapshots = solver.evaluator.add_file_handler(snapshots_dir,
                                                  sim_dt=T_F / NUM_SNAPSHOTS)
    snapshots.add_system(solver.state)

    # Main loop
    logger.info('Starting sim...')
    while solver.ok:
        solver.step(DT)
        curr_iter = solver.iteration

        if curr_iter % int((T_F / DT) / NUM_SNAPSHOTS) == 0:
            logger.info('Reached time %f out of %f',
                        solver.sim_time,
                        solver.stop_sim_time)

def default_problem(problem):
    problem.add_equation("dx(ux) + dz(uz) = 0")
    problem.add_equation("dt(rho) - rho0 * uz / H = 0")
    problem.add_equation("dt(ux) + dx(P) / rho0 = 0")
    problem.add_equation("dt(uz) + dz(P) / rho0 + rho * g / rho0 = 0")
    problem.add_bc("left(P) = 0", condition="nx == 0")
    problem.add_bc("left(uz) = cos(KX * x - omega * t)")

def get_omega(g, h, kx, kz):
    return np.sqrt((g / h) * kx**2 / (kx**2 + kz**2 + 0.25 / h**2))

def get_vph(g, h, kx, kz):
    norm = get_omega(g, h, kx, kz) / (kx**2 + kz**2)
    return norm * kz, norm * kz

def plot(setup_problem,
         set_ICs,
         XMAX,
         ZMAX,
         N_X,
         N_Z,
         T_F,
         DT,
         KX,
         KZ,
         H,
         RHO0,
         NUM_SNAPSHOTS,
         G,
         name=None):
    SAVE_FMT_STR = 't_%d.png'
    matplotlib.rcParams.update({'font.size': 6})
    snapshots_dir = SNAPSHOTS_DIR % name
    path = '{s}/{s}_s1'.format(s=snapshots_dir)
    filename = '{s}/{s}_s1/{s}_s1_p0.h5'.format(s=snapshots_dir)
    interp = 2
    plot_vars = ['uz', 'ux', 'rho', 'P']
    n_cols = 2
    n_rows = 2
    assert len(plot_vars) == n_cols * n_rows

    if not os.path.exists(snapshots_dir):
        raise ValueError('No snapshots dir "%s" found!' % snapshots_dir)

    solver, domain = get_solver(
        setup_problem,
        XMAX, ZMAX, N_X, N_Z, T_F, KX, KZ, H, RHO0, G)
    x = domain.grid(0, scales=interp)
    z = domain.grid(1, scales=interp)
    xmesh, zmesh = quad_mesh(x=x[:, 0], y=z[0])

    with h5py.File(filename, mode='r') as dat:
        sim_times = np.array(dat['scales']['sim_time'])

    state_vars = defaultdict(list)
    # we let the file close before trying to reopen it again in load
    for idx, time in enumerate(sim_times):
        solver.load_state(filename, idx)

        for varname in plot_vars:
            values = solver.state[varname]
            values.set_scales(interp, keep_data=True)
            state_vars[varname].append(np.copy(values['g']))

    for t_idx, sim_time in enumerate(sim_times):
        fig = plt.figure(dpi=200)

        for idx, var in enumerate(plot_vars):
            axes = fig.add_subplot(n_cols, n_rows, idx + 1, title=var)

            var_dat = np.array(state_vars[var])
            p = axes.pcolormesh(xmesh,
                                zmesh,
                                var_dat[t_idx].T,
                                vmin=var_dat.min(), vmax=var_dat.max())
            axes.axis(pad_limits(xmesh, zmesh))
            fig.colorbar(p, ax=axes)

        fig.suptitle('Config: %s (t=%.2f, kx=-2pi/H, kz=2pi/H)' % (name,
                                                           sim_time))
        fig.subplots_adjust(hspace=0.2, wspace=0.2)
        savefig = SAVE_FMT_STR % t_idx
        plt.savefig('%s/%s' % (path, savefig))
        print('Saved %s/%s' % (path, savefig))
        plt.close()
    os.system('ffmpeg -y -framerate 10 -i %s/%s %s.mp4' %
              (path, SAVE_FMT_STR, name))
