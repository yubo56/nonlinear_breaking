#!/usr/bin/env python
'''
helper function to run the shared stratification scenario. user just has to
specify BCs and ICs
'''
import logging
import os
from collections import defaultdict

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from dedalus import public as de
from dedalus.extras.flow_tools import CFL
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
               G,
               A,
               T0,
               NU):
    # Bases and domain
    x_basis = de.Fourier('x', N_X, interval=(0, XMAX), dealias=3/2)
    z_basis = de.Chebyshev('z', N_Z, interval=(0, ZMAX), dealias=3/2)
    domain = de.Domain([x_basis, z_basis], np.float64)
    z = domain.grid(1)

    problem = de.IVP(domain, variables=['P', 'rho', 'ux', 'uz', 'ux_z', 'uz_z'])
    problem.parameters['L'] = XMAX
    problem.parameters['g'] = G
    problem.parameters['H'] = H
    problem.parameters['A'] = A
    problem.parameters['KX'] = KX
    problem.parameters['KZ'] = KZ
    problem.parameters['NU'] = NU
    problem.parameters['RHO0'] = RHO0
    problem.parameters['T0'] = T0
    problem.parameters['omega'] = get_omega(G, H, KX, KZ)

    # rho0 stratification
    rho0 = domain.new_field()
    rho0.meta['x']['constant'] = True
    rho0['g'] = RHO0 * np.exp(-z / H)
    problem.parameters['rho0'] = rho0

    setup_problem(problem, domain)

    # Build solver
    solver = problem.build_solver(de.timesteppers.CNAB2)
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
                  A,
                  T0,
                  NU,
                  name=None,
                  USE_CFL=True,
                  **_):
    snapshots_dir = SNAPSHOTS_DIR % name
    try:
        os.makedirs(snapshots_dir)
    except FileExistsError:
        print('snapshots already exist, exiting...')
        return
    logger = logging.getLogger(name or __name__)

    solver, domain = get_solver(
        setup_problem,
        XMAX=XMAX, ZMAX=ZMAX, N_X=N_X, N_Z=N_Z, T_F=T_F, KX=KX, KZ=KZ, H=H,
        RHO0=RHO0, G=G, A=A, T0=T0, NU=NU)

    # Initial conditions
    set_ICs(solver, domain)

    cfl = CFL(solver, initial_dt=DT, cadence=10, max_dt=DT, threshold=0.10)
    cfl.add_velocities(('ux', 'uz'))
    snapshots = solver.evaluator.add_file_handler(snapshots_dir,
                                                  sim_dt=T_F / NUM_SNAPSHOTS)
    snapshots.add_system(solver.state)

    # Main loop
    logger.info('Starting sim...')
    while solver.ok:
        cfl_dt = cfl.compute_dt() if USE_CFL else DT
        solver.step(cfl_dt)
        curr_iter = solver.iteration

        if curr_iter % int((T_F / DT) / NUM_SNAPSHOTS) == 0:
            logger.info('Reached time %f out of %f, timestep %f vs max %f',
                        solver.sim_time,
                        solver.stop_sim_time,
                        cfl_dt, DT)

def default_problem(problem):
    problem.add_equation("dx(ux) + uz_z = 0")
    problem.add_equation("dt(rho) - rho0 * uz / H = 0")
    problem.add_equation(
        "dt(ux) + dx(P) / rho0 - NU * (dz(ux_z) + dx(dx(ux))) = 0")
    problem.add_equation(
        "dt(uz) + dz(P) / rho0 + rho * g / rho0 - NU * (dz(uz_z) + dx(dx(uz))) = 0")
    problem.add_equation("dz(ux) - ux_z = 0")
    problem.add_equation("dz(uz) - uz_z = 0")

    problem.add_bc("left(P) = 0", condition="nx == 0")
    problem.add_bc("left(uz) = A * cos(KX * x - omega * t)")
    problem.add_bc("left(ux) = -KZ / KX * A * cos(KX * x - omega * t)")

def get_omega(g, h, kx, kz):
    return np.sqrt((g / h) * kx**2 / (kx**2 + kz**2 + 0.25 / h**2))

def get_vph(g, h, kx, kz):
    norm = get_omega(g, h, kx, kz) / (kx**2 + kz**2)
    return norm * kz, norm * kz

def load(setup_problem,
         XMAX,
         ZMAX,
         N_X,
         N_Z,
         T_F,
         DT,
         KX,
         KZ,
         H,
         G,
         A,
         T0,
         NU,
         RHO0,
         INTERP_X,
         INTERP_Z,
         name=None,
         **_):
    dyn_vars = ['uz', 'ux', 'rho', 'P']
    snapshots_dir = SNAPSHOTS_DIR % name
    filename = '{s}/{s}_s1/{s}_s1_p0.h5'.format(s=snapshots_dir)

    if not os.path.exists(snapshots_dir):
        raise ValueError('No snapshots dir "%s" found!' % snapshots_dir)

    solver, domain = get_solver(
        setup_problem,
        XMAX=XMAX, ZMAX=ZMAX, N_X=N_X, N_Z=N_Z, T_F=T_F, KX=KX, KZ=KZ, H=H,
        RHO0=RHO0, G=G, A=A, T0=T0, NU=NU)
    z = domain.grid(1, scales=INTERP_Z)

    with h5py.File(filename, mode='r') as dat:
        sim_times = np.array(dat['scales']['sim_time'])
    # we let the file close before trying to reopen it again in load

    # load into state_vars
    state_vars = defaultdict(list)
    for idx in range(len(sim_times)):
        solver.load_state(filename, idx)

        for varname in dyn_vars:
            values = solver.state[varname]
            values.set_scales((INTERP_X, INTERP_Z), keep_data=True)
            state_vars[varname].append(np.copy(values['g']))
    # cast to np arrays
    for key in state_vars.keys():
        state_vars[key] = np.array(state_vars[key])

    if 'nonlin' not in name:
        state_vars['rho'] += RHO0 * np.exp(-z / H)
        state_vars['P'] += RHO0 * (np.exp(-z / H) - 1) * G * H
    state_vars['rho1'] = state_vars['rho'] - RHO0 * np.exp(-z / H)
    state_vars['P1'] = state_vars['P'] - RHO0 * (np.exp(-z / H) - 1) * G * H

    state_vars['E'] = state_vars['rho'] * \
                       (state_vars['ux']**2 + state_vars['uz']**2) / 2
    state_vars['F_z'] = state_vars['uz'] * (
        state_vars['rho'] * (state_vars['ux']**2 + state_vars['uz']**2)
        + state_vars['P'])
    return sim_times, domain, state_vars

def get_analytical_sponge(name, z_pts, t, A, T0, RHO0, OMEGA, KX, KZ, H):
    """ gets the analytical form of the variables for radiative BCs """
    uz_anal = A * np.exp(z_pts / (2 * H)) * np.cos(KZ * z_pts - OMEGA * t)
    rho0 = RHO0 * np.exp(-z_pts / H)
    analyticals = {
        'uz': uz_anal,
        'ux': -KZ / KX * uz_anal,
        'rho1': -rho0 * A / (H * OMEGA)* np.exp(z_pts / (2 * H)) \
            * np.sin(KZ * z_pts - OMEGA * t),
        'P1': -rho0 * OMEGA / KX**2 * KZ * uz_anal,
    }
    return analyticals[name]

def plot(setup_problem,
         XMAX,
         ZMAX,
         N_X,
         N_Z,
         T_F,
         DT,
         KX,
         KZ,
         H,
         G,
         A,
         T0,
         NU,
         RHO0,
         INTERP_X,
         INTERP_Z,
         name=None,
         **_):
    slice_suffix = '(x=0)' # slice suffix
    SAVE_FMT_STR = 't_%d.png'
    snapshots_dir = SNAPSHOTS_DIR % name
    path = '{s}/{s}_s1'.format(s=snapshots_dir)
    matplotlib.rcParams.update({'font.size': 6})
    plot_vars = ['uz', 'ux']
    z_vars = ['F_z', 'E'] # sum these over x
    slice_vars = ['%s%s' % (i, slice_suffix)
                  for i in ['uz', 'ux', 'rho1', 'P1']]
    n_cols = 4
    n_rows = 2
    plot_stride = 2

    if os.path.exists('%s.mp4' % name):
        print('%s.mp4 already exists, not regenerating' % name)
        return

    sim_times, domain, state_vars = load(setup_problem,
                                         XMAX=XMAX,
                                         ZMAX=ZMAX,
                                         N_X=N_X,
                                         N_Z=N_Z,
                                         T_F=T_F,
                                         DT=DT,
                                         KX=KX,
                                         KZ=KZ,
                                         H=H,
                                         G=G,
                                         A=A,
                                         T0=T0,
                                         NU=NU,
                                         RHO0=RHO0,
                                         INTERP_X=INTERP_X,
                                         INTERP_Z=INTERP_Z,
                                         name=name)

    x = domain.grid(0, scales=INTERP_X)
    z = domain.grid(1, scales=INTERP_Z)
    OMEGA = get_omega(G, H, KX, KZ)
    xmesh, zmesh = quad_mesh(x=x[:, 0], y=z[0])

    for var in z_vars:
        state_vars[var] = np.sum(state_vars[var], axis=1)
    for var in slice_vars:
        state_vars[var] = state_vars[var.replace(slice_suffix, '')][:, 0, :]

    for t_idx, sim_time in list(enumerate(sim_times))[::plot_stride]:
        fig = plt.figure(dpi=200)

        idx = 1
        for var in plot_vars:
            axes = fig.add_subplot(n_rows, n_cols, idx, title=var)

            var_dat = state_vars[var]
            p = axes.pcolormesh(xmesh,
                                zmesh,
                                var_dat[t_idx].T,
                                vmin=var_dat.min(), vmax=var_dat.max())
            axes.axis(pad_limits(xmesh, zmesh))
            cb = fig.colorbar(p, ax=axes)
            cb.ax.set_yticklabels(cb.ax.get_yticklabels(), rotation=30)
            plt.xticks(rotation=30)
            plt.yticks(rotation=30)
            idx += 1
        for var in z_vars + slice_vars:
            axes = fig.add_subplot(n_rows, n_cols, idx, title=var)
            var_dat = state_vars[var]
            z_pts = (zmesh[1:, 0] + zmesh[:-1, 0]) / 2
            p = axes.plot(var_dat[t_idx], z_pts)
            if slice_suffix in var:
                p = axes.plot(
                    get_analytical_sponge(
                        var.replace(slice_suffix, ''),
                        z_pts, sim_time, A, T0, RHO0, OMEGA, KX, KZ, H),
                    z_pts)
            plt.xticks(rotation=30)
            plt.yticks(rotation=30)
            axes.set_xlim(var_dat.min(), var_dat.max())
            idx += 1

        fig.suptitle('Config: %s (t=%.2f, kx=%.2f, kz=%.2f, omega=%.2f)' %
                     (name, sim_time, KX, KZ, OMEGA))
        fig.subplots_adjust(hspace=0.3, wspace=0.9)
        savefig = SAVE_FMT_STR % (t_idx // plot_stride)
        plt.savefig('%s/%s' % (path, savefig))
        print('Saved %s/%s' % (path, savefig))
        plt.close()
    os.system('ffmpeg -y -framerate 6 -i %s/%s %s.mp4' %
              (path, SAVE_FMT_STR, name))
