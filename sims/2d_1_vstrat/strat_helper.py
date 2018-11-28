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
plot_stride = 2

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
    z_top = z_bot + 2.5 * width
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
    filename = '{s}/{s}_s1.h5'.format(s=snapshots_dir)

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

def load(name, params, dyn_vars, plot_stride, start=0):
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
    for idx in range(len(sim_times))[start::plot_stride]:
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

    state_vars['rho'] += params['RHO0']
    state_vars['rho1'] = state_vars['rho'] - params['RHO0']
    state_vars['P1'] = state_vars['P']
    # compatibility
    state_vars['ux_z'] = np.gradient(state_vars['ux'], axis=-1) * \
        params['N_Z'] * params['INTERP_Z'] / params['ZMAX']

    state_vars['F_px'] = state_vars['rho'] * (state_vars['ux'] *
                                              state_vars['uz'])
    return sim_times[start::plot_stride], domain, state_vars

def plot(name, params):
    rank = CW.rank
    size = CW.size

    slice_suffix = '(x=0)'
    sum_suffix = '(mean)'
    sub_suffix = ' (- mean)'
    snapshots_dir = SNAPSHOTS_DIR % name
    path = '{s}/{s}_s1'.format(s=snapshots_dir)
    matplotlib.rcParams.update({'font.size': 6})
    N_X = params['N_X'] * params['INTERP_X']
    N_Z = params['N_Z'] * params['INTERP_Z']
    # z_b = N_Z // 4
    z_b = 0
    KX = params['KX']
    KZ = params['KZ']
    V_GZ = get_vgz(params['g'], params['H'], KX, KZ)

    # available cfgs:
    # plot_vars: 2D plot
    # c_vars: horizontally-summed vertical chebyshev components
    # f_vars: vertically-summed Fourier components
    # f2_vars: 2D plot w/ horizontal Fourier transform
    # slice_vars: sliced at x=0
    # z_vars: horizontally averaged
    # sub_vars: 2D plot, mean-subtracted
    def get_plot_vars(cfg):
        ''' unpacks above variables from cfg shorthand '''
        ret_vars = [
            cfg.get('plot_vars', []),
            [i + '_c' for i in cfg.get('c_vars', [])],
            [i + '_f' for i in cfg.get('f_vars', [])],
            cfg.get('f2_vars', []),
            [i + sum_suffix for i in cfg.get('z_vars', [])],
            [i + slice_suffix for i in cfg.get('slice_vars', [])],
            [i + sub_suffix for i in cfg.get('sub_vars', [])]]
        n_cols = cfg.get('n_cols', sum([len(arr) for arr in ret_vars]))
        n_rows = cfg.get('n_rows', 1)
        ret = [n_cols, n_rows, cfg['save_fmt_str']]
        ret.extend(ret_vars)
        return ret

    plot_cfgs = [
        {
            'save_fmt_str': 't_%03i.png',
            'z_vars': ['ux', 'F_px', 'ux_z'],
            'slice_vars': ['uz'],
            'sub_vars': ['ux'],
        },
        {
            'save_fmt_str': 'm_%03i.png',
            'plot_vars': ['ux', 'uz', 'rho1', 'P1'],
        },
    ]

    dyn_vars = ['uz', 'ux', 'rho', 'P']
    sim_times, domain, state_vars = load(name, params, dyn_vars, plot_stride)

    x = domain.grid(0, scales=params['INTERP_X'])
    z = domain.grid(1, scales=params['INTERP_Z'])[: , z_b:]
    xmesh, zmesh = quad_mesh(x=x[:, 0], y=z[0])
    x2mesh, z2mesh = quad_mesh(x=np.arange(params['N_X'] // 2), y=z[0])
    z_pts = np.array((zmesh[1:, 0] + zmesh[:-1, 0]) / 2)

    # preprocess
    for var in dyn_vars + ['F_px', 'ux_z']:
        state_vars[var + sum_suffix] = np.sum(state_vars[var], axis=1) / N_X

    for var in dyn_vars:
        state_vars[var + slice_suffix] = np.copy(state_vars[var][:, 0, :])

    for var in dyn_vars:
        # can't figure out how to numpy this together
        means = state_vars[var + sum_suffix]
        state_vars[var + sub_suffix] = np.copy(state_vars[var])
        for idx, _ in enumerate(state_vars[var + sub_suffix]):
            mean = state_vars[var + sum_suffix][idx]
            state_vars[var + sub_suffix][idx] -= np.tile(mean, (N_X, 1))

    for cfg in plot_cfgs:
        n_cols, n_rows, save_fmt_str, plot_vars, c_vars, f_vars,\
            f2_vars, slice_vars, z_vars, sub_vars = get_plot_vars(cfg)

        uz_est = params['F'] * get_uz_f_ratio(params)

        for t_idx, sim_time in list(enumerate(sim_times))[rank::size]:
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
                p = axes.plot(var_dat[t_idx],
                              z_pts,
                              linewidth=0.5)
                if var == 'uz%s' % slice_suffix:
                    p = axes.plot(
                        uz_est * np.ones(np.shape(z_pts)),
                        z_pts,
                        'orange',
                        linewidth=0.5)
                    p = axes.plot(
                        -uz_est * np.ones(np.shape(z_pts)),
                        z_pts,
                        'orange',
                        linewidth=0.5)

                if var == 'ux%s' % sum_suffix:
                    # mean flow = E[ux * uz] / V_GZ
                    p = axes.plot(
                        uz_est**2 * abs(KZ) / KX / (2 * abs(V_GZ)) *
                            np.ones(np.shape(z_pts)),
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
                'Config: %s (t=%.2f, kx(%.2f, %.2f), w=%.2f, v_gz=%.2f)' %
                (name, sim_time, KX, KZ, params['OMEGA'], V_GZ))
            fig.subplots_adjust(hspace=0.7, wspace=0.6)
            savefig = save_fmt_str % (t_idx)
            plt.savefig('%s/%s' % (snapshots_dir, savefig))
            logger.info('Saved %s/%s' % (snapshots_dir, savefig))
            plt.close()

def plot_front(name, params):
    ''' plots location of max Ri and flux @ that point, quadratically
    interpolated '''
    def get_quad_fit(x, fx):
        ''' a * x**2 + b * x + c = fx, return (a, b, c) '''
        return np.dot(
            np.linalg.inv(np.array([x**2, x, 0 * x + 1]).T),
            fx)

    N_X = params['N_X'] * params['INTERP_X']
    N_Z = params['N_Z'] * params['INTERP_Z']
    dyn_vars = ['uz', 'ux', 'rho', 'P']
    snapshots_dir = SNAPSHOTS_DIR % name
    logfile = '%s/data.log' % snapshots_dir

    sim_times = []
    front_pos = []
    ri_inv = []
    fluxes = []

    # load if exists
    if not os.path.exists(logfile):
        print('log file not found, generating')
        sim_times, domain, state_vars = load(
            name, params, dyn_vars, plot_stride, start=10)
        x = domain.grid(0, scales=params['INTERP_X'])
        z = domain.grid(1, scales=params['INTERP_Z'])
        xmesh, zmesh = quad_mesh(x=x[:, 0], y=z[0])
        z_pts = (zmesh[1:, 0] + zmesh[:-1, 0]) / 2

        ux_z = np.sum(state_vars['ux_z'], axis=1) / N_X
        F_px = np.sum(state_vars['F_px'], axis=1) / N_X
        for t_idx, sim_time in enumerate(sim_times):
            max_pos = np.argmax(ux_z[t_idx])

            ux_z_quad = get_quad_fit(z_pts[max_pos - 1:max_pos + 2],
                                     ux_z[t_idx][max_pos - 1:max_pos + 2])
            true_max = -ux_z_quad[1] / (2 * ux_z_quad[0]) # -b/2a
            front_pos.append(true_max)
            ri_inv.append((ux_z_quad[0] * true_max**2
                           + ux_z_quad[1] * true_max
                           + ux_z_quad[2]) / (params['g'] / params['H']))

            fluxes_quad = get_quad_fit(z_pts[max_pos - 1:max_pos + 2],
                                       F_px[t_idx][max_pos - 1:max_pos + 2])
            fluxes.append((fluxes_quad[0] * true_max**2
                           + fluxes_quad[1] * true_max
                           + fluxes_quad[2])* 2)
        with open(logfile, 'w') as data:
            data.write(repr(sim_times.tolist()))
            data.write('\n')
            data.write(repr(front_pos))
            data.write('\n')
            data.write(repr(ri_inv))
            data.write('\n')
            data.write(repr(fluxes))

    else:
        print('data.log found, loading')
        with open(logfile) as data:
            sim_times = eval(data.readline())
            front_pos = eval(data.readline())
            ri_inv = eval(data.readline())
            fluxes = eval(data.readline())
    plt.plot(sim_times, front_pos)
    plt.ylabel('Front Position')
    plt.xlabel('Time')
    plt.title(name)
    plt.savefig('%s/front.png' % snapshots_dir, dpi=200)
    plt.clf()

    plt.plot(sim_times, ri_inv)
    plt.ylabel('1/Ri')
    plt.xlabel('Time')
    plt.title(name)
    plt.savefig('%s/f_ri.png' % snapshots_dir, dpi=200)
    plt.clf()

    plt.plot(sim_times, fluxes)
    plt.ylabel('F_px')
    plt.xlabel('Time')
    plt.title(name)
    plt.locator_params(nbins=3)
    plt.savefig('%s/fluxes.png' % snapshots_dir, dpi=200)
    plt.clf()
