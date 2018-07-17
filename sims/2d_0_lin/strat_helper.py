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
from dedalus.tools import post
from dedalus.extras.flow_tools import CFL
from dedalus.extras.plot_tools import quad_mesh, pad_limits

SNAPSHOTS_DIR = 'snapshots_%s'

def get_omega(g, h, kx, kz):
    return np.sqrt((g / h) * kx**2 / (kx**2 + kz**2 + 0.25 / h**2))

def get_vg(g, h, kx, kz):
    norm = get_omega(g, h, kx, kz) / (kx**2 + kz**2) * (kz / kx)
    return norm * kx, norm * kz

def zero_ic(solver, domain, params):
    pass

def get_solver(params):
    ''' sets up solver '''
    x_basis = de.Fourier('x',
                         params['N_X'],
                         interval=(0, params['XMAX']),
                         dealias=3/2)
    z_basis = de.Chebyshev('z',
                           params['N_Z'],
                           interval=(0, params['ZMAX']),
                           dealias=3/2)
    domain = de.Domain([x_basis, z_basis], np.float64)
    z = domain.grid(1)

    problem = de.IVP(domain, variables=['P', 'rho', 'ux', 'uz'])
    problem.parameters.update(params)

    # rho0 stratification
    rho0 = domain.new_field()
    rho0.meta['x']['constant'] = True
    rho0['g'] = params['RHO0'] * np.exp(-z / params['H'])
    problem.parameters['rho0'] = rho0

    problem.substitutions['sponge'] = 'SPONGE_STRENGTH * 0.5 * ' +\
        '(2 + tanh((z - SPONGE_HIGH) / (0.3 * (ZMAX - SPONGE_HIGH))) - ' +\
        'tanh((z - SPONGE_LOW) / (0.3 * (SPONGE_LOW))))'
    problem.add_equation('dx(ux) + dz(uz) = 0')
    problem.add_equation(
        'dt(rho) - rho0 * uz / H' +
        '= - sponge * rho +' +
        'F * exp(-(z - Z0)**2 / (2 * S**2)) *' +
            'cos(KX * x - OMEGA * t)')
    problem.add_equation(
        'dt(ux) + dx(P) / rho0' +
        '= - sponge * ux')
    problem.add_equation(
        'dt(uz) + dz(P) / rho0 + rho * g / rho0' +
        '= - sponge * uz')

    problem.add_bc('left(uz) = 0')
    problem.add_bc('right(uz) = 0', condition = 'nx != 0')
    problem.add_bc('right(P) = 0', condition = 'nx == 0')

    # Build solver
    solver = problem.build_solver(de.timesteppers.RK443)
    solver.stop_sim_time = params['T_F']
    solver.stop_wall_time = np.inf
    solver.stop_iteration = np.inf
    return solver, domain

###
### ENTRY POINTS
###

def run_strat_sim(set_ICs, name, params):
    snapshots_dir = SNAPSHOTS_DIR % name
    logger = logging.getLogger(name)

    solver, domain = get_solver(params)

    # Initial conditions
    set_ICs(solver, domain, params)

    cfl = CFL(solver,
              initial_dt=params['DT'],
              cadence=10,
              max_dt=params['DT'],
              threshold=0.10)
    cfl.add_velocities(('ux', 'uz'))
    snapshots = solver.evaluator.add_file_handler(
        snapshots_dir,
        sim_dt=params['T_F'] / params['NUM_SNAPSHOTS'])
    snapshots.add_system(solver.state)

    # Main loop
    logger.info('Starting sim...')
    while solver.ok:
        cfl_dt = cfl.compute_dt() if params.get('USE_CFL') else params['DT']
        solver.step(cfl_dt)
        curr_iter = solver.iteration

        if curr_iter % int((params['T_F'] / params['DT']) /
                           params['NUM_SNAPSHOTS']) == 0:
            logger.info('Reached time %f out of %f, timestep %f vs max %f',
                        solver.sim_time,
                        solver.stop_sim_time,
                        cfl_dt,
                        params['DT'])

def load(name, params):
    dyn_vars = ['uz', 'ux', 'rho', 'P']
    snapshots_dir = SNAPSHOTS_DIR % name
    filename = '{s}/{s}_s1.h5'.format(s=snapshots_dir)

    post.merge_analysis(snapshots_dir, cleanup=False)

    solver, domain = get_solver(params)
    z = domain.grid(1, scales=params['INTERP_Z'])

    with h5py.File(filename, mode='r') as dat:
        sim_times = np.array(dat['scales']['sim_time'])
    # we let the file close before trying to reopen it again in load

    # load into state_vars
    state_vars = defaultdict(list)
    for idx in range(len(sim_times)):
        solver.load_state(filename, idx)

        for varname in dyn_vars:
            values = solver.state[varname]
            values.set_scales((params['INTERP_X'], params['INTERP_Z']),
                              keep_data=True)
            state_vars[varname].append(np.copy(values['g']))
    # cast to np arrays
    for key in state_vars.keys():
        state_vars[key] = np.array(state_vars[key])

    state_vars['rho'] += params['RHO0'] * np.exp(-z / params['H'])
    state_vars['P'] += params['RHO0'] * (np.exp(-z / params['H']) - 1) *\
        params['g'] * params['H']
    state_vars['rho1'] = state_vars['rho'] - params['RHO0'] *\
        np.exp(-z / params['H'])
    state_vars['P1'] = state_vars['P'] -\
        params['RHO0'] * (np.exp(-z / params['H']) - 1) *\
        params['g'] * params['H']

    state_vars['E'] = params['RHO0'] * np.exp(-z / params['H']) * \
                       (state_vars['ux']**2 + state_vars['uz']**2) / 2
    state_vars['F_z'] = state_vars['uz'] * (
        state_vars['rho'] * (state_vars['ux']**2 + state_vars['uz']**2)
        + state_vars['P'])
    return sim_times, domain, state_vars

def plot(name, params):
    slice_suffix = '(x=0)' # slice suffix
    SAVE_FMT_STR = 't_%d.png'
    snapshots_dir = SNAPSHOTS_DIR % name
    path = '{s}/{s}_s1'.format(s=snapshots_dir)
    matplotlib.rcParams.update({'font.size': 6})
    plot_vars = ['uz', 'ux']
    z_vars = ['F_z', 'E'] # sum these over x
    slice_vars = ['%s%s' % (i, slice_suffix) for i in ['uz', 'ux']]
    n_cols = 3
    n_rows = 2
    plot_stride = 1

    if os.path.exists('%s.mp4' % name):
        print('%s.mp4 already exists, not regenerating' % name)
        return

    sim_times, domain, state_vars = load(name, params)

    x = domain.grid(0, scales=params['INTERP_X'])
    z = domain.grid(1, scales=params['INTERP_Z'])
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
                                var_dat[t_idx].T)
                                # vmin=var_dat.min(), vmax=var_dat.max())
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

            plt.xticks(rotation=30)
            plt.yticks(rotation=30)
            xlims = [var_dat.min(), var_dat.max()]
            axes.set_xlim(*xlims)
            p = axes.plot(xlims, [params['SPONGE_LOW']] * len(xlims), 'r.')
            p = axes.plot(xlims, [params['SPONGE_HIGH']] * len(xlims), 'r.')
            p = axes.plot(xlims, [params['Z0']] * len(xlims), 'b--')
            idx += 1

        fig.suptitle(
            'Config: %s (t=%.2f, kx=%.2f, kz=%.2f, omega=%.2f)' %
            (name, sim_time, params['KX'], params['KZ'], params['OMEGA']))
        fig.subplots_adjust(hspace=0.7, wspace=0.6)
        savefig = SAVE_FMT_STR % (t_idx // plot_stride)
        plt.savefig('%s/%s' % (snapshots_dir, savefig))
        print('Saved %s/%s' % (snapshots_dir, savefig))
        plt.close()
    os.system('ffmpeg -y -framerate 12 -i %s/%s %s.mp4' %
              (snapshots_dir, SAVE_FMT_STR, name))
