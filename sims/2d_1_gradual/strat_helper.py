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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')

from dedalus import public as de
from dedalus.tools import post
from dedalus.extras.flow_tools import CFL
from dedalus.extras.plot_tools import quad_mesh, pad_limits

SNAPSHOTS_DIR = 'snapshots_%s'

def get_omega(g, h, kx, kz):
    return np.sqrt((g / h) * kx**2 / (kx**2 + kz**2 + 0.25 / h**2))

def get_vgz(g, h, kx, kz):
    return get_omega(g, h, kx, kz) / (kx**2 + kz**2 + 0.25 / h**2) * kz

def zero_ic(solver, domain, params):
    pass

def get_uz_f_ratio(params):
    return (np.sqrt(2 * np.pi) * params['S'] * params['g'] *
            params['KX']**2) * np.exp(-1/2) / (
                2 * params['RHO0'] * np.exp(-params['Z0'] / params['H'])
                * params['OMEGA']**2 * params['KZ'])

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

    problem = de.IVP(domain, variables=['P', 'rho', 'ux', 'uz',
                                        'ux_z', 'uz_z', 'rho_z',
                                        ])
    problem.parameters.update(params)

    # rho0 stratification
    rho0 = domain.new_field()
    rho0.meta['x']['constant'] = True
    rho0['g'] = params['RHO0'] * np.exp(-z / params['H'])
    problem.parameters['rho0'] = rho0

    problem.substitutions['sponge'] = 'SPONGE_STRENGTH * 0.5 * ' +\
        '(2 + tanh((z - SPONGE_HIGH) / (SPONGE_WIDTH * (ZMAX - SPONGE_HIGH))) - ' +\
        'tanh((z - SPONGE_LOW) / (SPONGE_WIDTH * (SPONGE_LOW))))'
    problem.add_equation('dx(ux) + uz_z = 0')
    problem.add_equation(
        'dt(rho) - rho0 * uz / H' +
        '- NU * (dx(dx(rho)) + dz(rho_z))' +
        '= - sponge * rho - ux * dx(rho) - uz * dz(rho) +' +
        '(t / 500)**2 / ((t / 500)**2 + 1) * F * exp(-(z - Z0)**2 / (2 * S**2)) *' +
            'cos(KX * x - OMEGA * t)')
    problem.add_equation(
        'dt(ux) + dx(P) / rho0' +
        '- NU * (dx(dx(ux)) + dz(ux_z))' +
        '= - sponge * ux - ux * dx(ux) - uz * dz(ux)')
    problem.add_equation(
        'dt(uz) + dz(P) / rho0 + rho * g / rho0' +
        '- NU * (dx(dx(uz)) + dz(uz_z))' +
        '= - sponge * uz - ux * dx(uz) - uz * dz(uz)')
    problem.add_equation('dz(ux) - ux_z = 0')
    problem.add_equation('dz(uz) - uz_z = 0')
    problem.add_equation('dz(rho) - rho_z = 0')


    problem.add_bc('left(uz) = 0')
    problem.add_bc('left(ux) = 0')
    problem.add_bc('right(ux) = 0')
    problem.add_bc('right(P) = 0')
    problem.add_bc('right(rho) = 0')
    problem.add_bc('left(rho_z) = 0')

    # Build solver
    solver = problem.build_solver(de.timesteppers.RK443)
    solver.stop_sim_time = params['T_F']
    solver.stop_wall_time = np.inf
    solver.stop_iteration = np.inf
    return solver, domain

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
              safety=0.5,
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
            state_vars['%s_c' % varname].append(np.copy(np.abs(values['c'])))
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
    slice_suffix = '(x=0)'
    SAVE_FMT_STR = 't_%d.png'
    snapshots_dir = SNAPSHOTS_DIR % name
    path = '{s}/{s}_s1'.format(s=snapshots_dir)
    matplotlib.rcParams.update({'font.size': 6})
    plot_vars = ['uz']
    c_vars = ['uz_c']
    f_vars = ['uz_f']
    # z_vars = ['F_z', 'E'] # sum these over x
    z_vars = []
    slice_vars = ['%s%s' % (i, slice_suffix) for i in ['uz']]
    n_cols = 4
    n_rows = 1
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

    uz_est = params['F'] * get_uz_f_ratio(params)

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
            p = axes.plot(var_dat[t_idx],
                          z_pts,
                          linewidth=0.5)
            if var == 'uz%s' % slice_suffix:
                p = axes.plot(
                    uz_est * np.exp((z_pts - params['Z0']) / (2 * params['H'])),
                    z_pts,
                    'orange',
                    linewidth=0.5)
                p = axes.plot(
                    -uz_est * np.exp((z_pts - params['Z0']) / (2 * params['H'])),
                    z_pts,
                    'orange',
                    linewidth=0.5)

            plt.xticks(rotation=30)
            plt.yticks(rotation=30)
            xlims = [var_dat[t_idx].min(), var_dat[t_idx].max()]
            axes.set_xlim(*xlims)
            p = axes.plot(xlims,
                          [params['SPONGE_LOW']] * len(xlims),
                          'r:',
                          linewidth=0.5)
            p = axes.plot(xlims,
                          [params['SPONGE_HIGH']] * len(xlims),
                          'r:',
                          linewidth=0.5)
            p = axes.plot(xlims,
                          [params['Z0'] + 3 * params['S']] * len(xlims),
                          'b--',
                          linewidth=0.5)
            p = axes.plot(xlims,
                          [params['Z0'] - 3 * params['S']] * len(xlims),
                          'b--',
                          linewidth=0.5)
            idx += 1
        for var in c_vars:
            axes = fig.add_subplot(n_rows,
                                   n_cols,
                                   idx,
                                   title='%s (kx=kx_d)' % var)
            var_dat = state_vars[var]
            kx_idx = round(params['KX'] / (2 * np.pi / params['XMAX']))
            p = axes.semilogx(var_dat[t_idx][kx_idx],
                              range(len(var_dat[t_idx][kx_idx])),
                              linewidth=0.5)
            idx += 1

        for var in f_vars:
            axes = fig.add_subplot(n_rows,
                                   n_cols,
                                   idx,
                                   title='%s (Cheb. summed)' % var)
            var_dat = state_vars[var.replace('_f', '_c')]
            summed_dat = np.sum(np.abs(var_dat[t_idx]), 1)
            p = axes.semilogx(summed_dat, range(len(summed_dat)), linewidth=0.5)
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
