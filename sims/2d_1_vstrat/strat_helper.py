#!/usr/bin/env python
'''
helper function to run the shared stratification scenario. user just has to
specify BCs and ICs
'''
import logging
logger = logging.getLogger()

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
from dedalus.extras.flow_tools import CFL, GlobalFlowProperty
from dedalus.extras.plot_tools import quad_mesh, pad_limits
from mpi4py import MPI
CW = MPI.COMM_WORLD

SNAPSHOTS_DIR = 'snapshots_%s'

def get_omega(g, h, kx, kz):
    return np.sqrt((g / h) * kx**2 / (kx**2 + kz**2 + 0.25 / h**2))

def get_vgz(g, h, kx, kz):
    return get_omega(g, h, kx, kz) / (kx**2 + kz**2 + 0.25 / h**2) * kz

def set_ic(solver, domain, params):
    ux = solver.state['ux']
    z = domain.grid(1)

    # turns on at Z0 + ZMAX / 2 w/ width 2 * lambda_z, turns off at sponge zone
    zmax = params['ZMAX']
    KZ = params['KZ']
    z_bot = (params['Z0'] + params['ZMAX']) / 2
    width = abs(np.pi / KZ)
    z_top = z_bot + 3 * width
    ux['g'] = params['OMEGA'] / params['KX'] * params['UZ0_COEFF'] * (
        np.tanh((z - z_bot) / width) -
        np.tanh((z - z_top) / (0.3 * (zmax - z_top)))) / 2

def get_uz_f_ratio(params):
    return (np.sqrt(2 * np.pi) * params['S'] * params['g'] *
            params['KX']**2) * np.exp(-1/2) / (
                2 * params['RHO0'] * params['OMEGA']**2 * params['KZ'])

def get_solver(params):
    ''' sets up solver '''
    x_basis = de.Fourier('x',
                         params['N_X'],
                         interval=(0, params['XMAX']),
                         dealias=3/2)
    z_basis = de.Fourier('z',
                         params['N_Z'],
                         interval=(0, params['ZMAX']),
                         dealias=3/2)
    domain = de.Domain([x_basis, z_basis], np.float64)

    problem = de.IVP(domain, variables=['P', 'rho', 'ux', 'uz'])
    problem.parameters.update(params)

    problem.substitutions['sponge'] = 'SPONGE_STRENGTH * 0.5 * ' +\
        '(2 + tanh((z - SPONGE_HIGH) / (SPONGE_WIDTH * (ZMAX - SPONGE_HIGH))) - ' +\
        'tanh((z - SPONGE_LOW) / (SPONGE_WIDTH * (SPONGE_LOW))))'
    problem.add_equation('dx(ux) + dz(uz) = 0', condition='nx != 0 or nz != 0')
    problem.add_equation(
        'dt(rho) - RHO0 * uz / H' +
        '- NU * (N_Z/N_X)**6 * dx(dx(dx(dx(dx(dx(rho))))))' +
        '- NU * dz(dz(dz(dz(dz(dz(rho))))))' +
        '= -sponge * rho' +
        '- (ux * dx(rho) + uz * dz(rho))' +
        '+ F * exp(-(z - Z0)**2 / (2 * S**2)) *cos(KX * x - OMEGA * t)')
    problem.add_equation(
        'dt(ux) + dx(P) / RHO0' +
        '- NU * (N_Z/N_X)**6 * dx(dx(dx(dx(dx(dx(ux))))))' +
        '- NU * dz(dz(dz(dz(dz(dz(ux))))))' +
        '= - sponge * ux' +
        '- (ux * dx(ux) + uz * dz(ux))')
    problem.add_equation(
        'dt(uz) + dz(P) / RHO0 + rho * g / RHO0' +
        '- NU * (N_Z/N_X)**6 * dx(dx(dx(dx(dx(dx(uz))))))' +
        '- NU * dz(dz(dz(dz(dz(dz(uz))))))' +
        '= -sponge * uz' +
        '- (ux * dx(uz) + uz * dz(uz))')
    problem.add_equation('P = 0', condition='nx == 0 and nz == 0')

    # Build solver
    solver = problem.build_solver(de.timesteppers.RK443)
    solver.stop_sim_time = params['T_F']
    solver.stop_wall_time = np.inf
    solver.stop_iteration = np.inf
    return solver, domain

def run_strat_sim(set_ICs, name, params):
    snapshots_dir = SNAPSHOTS_DIR % name
    if os.path.exists(snapshots_dir):
        print('%s already ran, not rerunning' % name)
        return

    logger = logging.getLogger(name)

    solver, domain = get_solver(params)

    # Initial conditions
    set_ICs(solver, domain, params)

    cfl = CFL(solver,
              initial_dt=params['DT'],
              cadence=10,
              max_dt=params['DT'],
              min_dt=0.01,
              safety=0.5,
              threshold=0.10)
    cfl.add_velocities(('ux', 'uz'))
    cfl.add_frequency(params['DT'])
    snapshots = solver.evaluator.add_file_handler(
        snapshots_dir,
        sim_dt=params['T_F'] / params['NUM_SNAPSHOTS'])
    snapshots.add_system(solver.state)

    # Flow properties
    flow = GlobalFlowProperty(solver, cadence=10)
    flow.add_property('sqrt((ux / NU)**2 + (uz / NU)**2)', name='Re')

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
            logger.info('Max Re = %f' %flow.max('Re'))

def merge(name):
    snapshots_dir = SNAPSHOTS_DIR % name
    filename = '{s}/{s}_s1.h5'.format(s=snapshots_dir)

    if not os.path.exists(filename):
        post.merge_analysis(snapshots_dir)

def load(name, params):
    dyn_vars = ['uz', 'ux', 'rho', 'P']
    snapshots_dir = SNAPSHOTS_DIR % name
    filename = '{s}/{s}_s1.h5'.format(s=snapshots_dir)

    merge(name)

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
    sum_suffix = '(mean)'
    sub_suffix = ' (- mean)'
    SAVE_FMT_STR = 't_%03i.png'
    snapshots_dir = SNAPSHOTS_DIR % name
    path = '{s}/{s}_s1'.format(s=snapshots_dir)
    matplotlib.rcParams.update({'font.size': 6})
    plot_vars = ['ux']
    # c_vars = ['uz_c']
    # f_vars = ['uz_f']
    # f2_vars = ['ux']
    z_vars = ['%s%s' % (i, sum_suffix) for i in ['ux']] # sum these over x
    slice_vars = ['%s%s' % (i, slice_suffix) for i in ['uz']]
    sub_vars = ['%s%s' % (i, sub_suffix) for i in ['ux']]
    # plot_vars = []
    c_vars = []
    f_vars = []
    f2_vars = []
    # z_vars = []
    # slice_vars = []
    # sub_vars = []
    n_cols = 4
    n_rows = 1
    plot_stride = 1
    N_X = params['N_X'] * params['INTERP_X']
    N_Z = params['N_Z'] * params['INTERP_Z']
    # z_b = N_Z // 4
    z_b = 0

    if os.path.exists('%s.mp4' % name):
        print('%s.mp4 already exists, not regenerating' % name)
        return

    sim_times, domain, state_vars = load(name, params)

    x = domain.grid(0, scales=params['INTERP_X'])
    z = domain.grid(1, scales=params['INTERP_Z'])[: , z_b:]
    xmesh, zmesh = quad_mesh(x=x[:, 0], y=z[0])
    x2mesh, z2mesh = quad_mesh(x=np.arange(params['N_X'] // 2), y=z[0])

    for var in z_vars:
        state_vars[var] = np.sum(state_vars[var.replace(sum_suffix, '')],
                                 axis=1) / N_X
    for var in slice_vars:
        state_vars[var] = np.copy(
            state_vars[var.replace(slice_suffix, '')][:, 0, :])

    for var in sub_vars:
        # can't figure out how to numpy this together
        means = state_vars[var.replace(sub_suffix, sum_suffix)]
        state_vars[var] = np.copy(state_vars[var.replace(sub_suffix, '')])
        for idx, _ in enumerate(state_vars[var]):
            mean = state_vars[var.replace(sub_suffix, sum_suffix)][idx]
            state_vars[var][idx] -= np.tile(mean, (N_X, 1))

    uz_est = params['F'] * get_uz_f_ratio(params)

    for t_idx, sim_time in list(enumerate(sim_times))[::plot_stride]:
        fig = plt.figure(dpi=200)

        idx = 1
        for var in plot_vars + sub_vars:
            axes = fig.add_subplot(n_rows, n_cols, idx, title=var)

            var_dat = state_vars[var][:, : , z_b:]
            p = axes.pcolormesh(xmesh,
                                zmesh,
                                var_dat[t_idx].T,
                                vmin=var_dat.min(), vmax=var_dat.max())
            axes.axis(pad_limits(xmesh, zmesh))
            cb = fig.colorbar(p, ax=axes)
            plt.xticks(rotation=30)
            plt.yticks(rotation=30)
            idx += 1

        for var in f2_vars:
            axes = fig.add_subplot(n_rows, n_cols, idx,
                                   title='log %s (x-FT)' % var)

            var_dat = state_vars[var][:, : , z_b:]
            var_dat_t = np.fft.fft(var_dat[t_idx], axis=0)
            var_dat_shaped = np.log(np.abs(
                2 * var_dat_t.real[:params['N_X'] // 2, :]))
            p = axes.pcolormesh(x2mesh,
                                z2mesh,
                                var_dat_shaped.T,
                                vmin=var_dat.min(), vmax=var_dat.max())
            axes.axis(pad_limits(x2mesh, z2mesh))
            cb = fig.colorbar(p, ax=axes)
            plt.xticks(rotation=30)
            plt.yticks(rotation=30)
            idx += 1

        for var in z_vars + slice_vars:
            axes = fig.add_subplot(n_rows, n_cols, idx, title=var)
            var_dat = state_vars[var][:, z_b:]
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
            xlims = [var_dat.min(), var_dat.max()]
            axes.set_xlim(*xlims)
            axes.set_ylim(z_pts.min(), z_pts.max())
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
            p = axes.plot(xlims,
                          [(params['Z0'] + params['ZMAX']) / 2] * len(xlims),
                          'g--',
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
    # os.system('ffmpeg -y -framerate 12 -i %s/%s %s.mp4' %
    #           (snapshots_dir, SAVE_FMT_STR, name))
