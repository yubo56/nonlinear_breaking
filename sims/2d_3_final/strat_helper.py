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
FILENAME_EXPR = '{s}/{s}_s{idx}.h5'
plot_stride = 2

def get_omega(g, h, kx, kz):
    return np.sqrt((g / h) * kx**2 / (kx**2 + kz**2 + 0.25 / h**2))

def get_vgz(g, h, kx, kz):
    return get_omega(g, h, kx, kz) / (kx**2 + kz**2 + 0.25 / h**2) * kz

def set_ic(name, solver, domain, params):
    snapshots_dir = SNAPSHOTS_DIR % name
    filename = FILENAME_EXPR.format(s=snapshots_dir, idx=1)

    if not os.path.exists(snapshots_dir):
        print('No snapshots found, no IC loaded')
        return 0, params['DT']

    # snapshots exist, merge if need and then load
    print('Attempting to load snapshots')
    write, dt = solver.load_state(filename, -1)
    print('Loaded snapshots')
    return write, dt

def get_uz_f_ratio(params):
    return (np.sqrt(2 * np.pi) * params['S'] * params['g'] *
            params['KX']**2) * np.exp(-params['S']**2 * params['KZ']**2/2) / (
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

    problem = de.IVP(domain, variables=['W', 'U', 'ux', 'uz',
                                        'ux_z', 'uz_z', 'U_z',
                                        ])
    problem.parameters.update(params)

    problem.substitutions['sponge'] = 'SPONGE_STRENGTH * 0.5 * ' +\
        '(2 + tanh((z - SPONGE_HIGH) / (SPONGE_WIDTH * (ZMAX - SPONGE_HIGH))) - ' +\
        'tanh((z - SPONGE_LOW) / (SPONGE_WIDTH * (SPONGE_LOW))))'
    problem.substitutions['rho0'] = 'RHO0 * exp(-z / H)'
    problem.add_equation('dx(ux) + uz_z = 0')
    problem.add_equation(
        'dt(U) - uz / H' +
        '- NU * (dx(dx(U)) + dz(U_z) - 2 * U_z / H)' +
        '= - sponge * U' +
        '- (ux * dx(U) + uz * dz(U))' +
        '+ NU * (dx(U) * dx(U) + U_z * U_z)' +
        '+ F * exp(-(z - Z0)**2 / (2 * S**2) + Z0 / H) *' +
            'cos(KX * x - OMEGA * t)')
    problem.add_equation(
        'dt(ux) + dx(W) + (g * H) * dx(U)' +
        '- (NU * dx(dx(ux)) + NU * dz(ux_z))' +
        '+ NU * dz(ux) / H'
        '= - sponge * ux - (ux * dx(ux) + uz * dz(ux))' +
        '- NU * (dx(U) * dx(ux) + U_z * ux_z)' +
        '+ NU * ux * (dx(dx(U)) + dz(U_z))' +
        '+ NU * ux * (dx(U) * dx(U) + U_z * U_z)' +
        '- 2 * NU * ux * U_z / H' +
        # '+ NU * ux * (1 - exp(-U)) / H**2' +
        '+ NU * ux * U / H**2' +
        '- W * dx(U)')
    problem.add_equation(
        'dt(uz) + dz(W) + (g * H) * dz(U) - W/H' +
        '- (NU * dx(dx(uz)) + NU * dz(uz_z))' +
        '+ NU * dz(uz) / H'
        '= - sponge * uz - (ux * dx(uz) + uz * dz(uz))' +
        '- NU * (dx(U) * dx(uz) + U_z * uz_z)' +
        '+ NU * uz * (dx(dx(U)) + dz(U_z))' +
        '+ NU * uz * (dx(U) * dx(U) + U_z * U_z)' +
        '- 2 * NU * uz * U_z / H' +
        # '+ NU * uz * (1 - exp(-U)) / H**2' +
        '+ NU * uz * U / H**2' +
        '- W * dz(U)')
    problem.add_equation('dz(ux) - ux_z = 0')
    problem.add_equation('dz(uz) - uz_z = 0')
    problem.add_equation('dz(U) - U_z = 0')


    problem.add_bc('right(uz) = 0')
    problem.add_bc('left(W) = 0', condition='nx == 0')
    problem.add_bc('left(uz) = 0', condition='nx != 0')
    problem.add_bc('left(ux) = 0')
    problem.add_bc('right(ux) = 0')
    problem.add_bc('right(U) = 0')
    problem.add_bc('left(U) = 0')

    # Build solver
    solver = problem.build_solver(de.timesteppers.RK443)
    solver.stop_sim_time = params['T_F']
    solver.stop_wall_time = np.inf
    solver.stop_iteration = np.inf
    return solver, domain

def run_strat_sim(set_ICs, name, params):
    snapshots_dir = SNAPSHOTS_DIR % name

    solver, domain = get_solver(params)

    # Initial conditions
    dt = params['DT']
    _, dt = set_ICs(name, solver, domain, params)

    cfl = CFL(solver,
              initial_dt=dt,
              cadence=5,
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
    flow.add_property('integ(ux_z, "x") / (XMAX * g / H)', name='Ri_inv')

    # Main loop
    logger.info('Starting sim...')
    while solver.ok:
        cfl_dt = cfl.compute_dt() if params.get('USE_CFL') else params['DT']
        # if cfl_dt < params['DT'] / 8: # small step sizes if strongly cfl limited
        #     cfl_dt /= 2
        solver.step(cfl_dt)
        curr_iter = solver.iteration

        if curr_iter % int((params['T_F'] / params['DT']) /
                           params['NUM_SNAPSHOTS']) == 0:
            logger.info('Reached time %f out of %f, timestep %f vs max %f',
                        solver.sim_time,
                        solver.stop_sim_time,
                        cfl_dt,
                        params['DT'])
            logger.info('Max Re = %e, Max Ri_inv = %f' % (flow.max('Re'),
                                                          flow.max('Ri_inv')))

def merge(name):
    snapshots_dir = SNAPSHOTS_DIR % name
    filename = FILENAME_EXPR.format(s=snapshots_dir, idx=1)
    if not os.path.exists(filename):
        post.merge_analysis(snapshots_dir)

def load(name, params, dyn_vars, plot_stride, start=0):
    snapshots_dir = SNAPSHOTS_DIR % name
    merge(name)

    solver, domain = get_solver(params)
    z = domain.grid(1, scales=params['INTERP_Z'])

    i = 1
    filename = FILENAME_EXPR.format(s=snapshots_dir, idx=i)
    total_sim_times = []
    state_vars = defaultdict(list)

    while os.path.exists(filename):
        print('Loading %s' % filename)
        with h5py.File(filename, mode='r') as dat:
            sim_times = np.array(dat['scales']['sim_time'])
        # we let the file close before trying to reopen it again in load

        # load into state_vars
        for idx in range(len(sim_times))[start::plot_stride]:
            solver.load_state(filename, idx)

            for varname in dyn_vars:
                values = solver.state[varname]
                values.set_scales((params['INTERP_X'], params['INTERP_Z']),
                                  keep_data=True)
                state_vars[varname].append(np.copy(values['g']))
                state_vars['%s_c' % varname].append(np.copy(np.abs(values['c'])))

        total_sim_times.extend(sim_times)
        i += 1
        filename = FILENAME_EXPR.format(s=snapshots_dir, idx=i)

    # cast to np arrays
    for key in state_vars.keys():
        state_vars[key] = np.array(state_vars[key])

    state_vars['F_px'] = params['RHO0']\
        * np.exp(-z/ params['H'])\
        * (state_vars['ux'] * state_vars['uz'])
    return np.array(total_sim_times[start::plot_stride]), domain, state_vars

def plot(name, params):
    rank = CW.rank
    size = CW.size

    slice_suffix = '(x=0)'
    mean_suffix = '(mean)'
    sub_suffix = ' (- mean)'
    snapshots_dir = SNAPSHOTS_DIR % name
    matplotlib.rcParams.update({'font.size': 6})
    N_X = params['N_X'] * params['INTERP_X']
    N_Z = params['N_Z'] * params['INTERP_Z']
    KX = params['KX']
    KZ = params['KZ']
    V_GZ = get_vgz(params['g'], params['H'], KX, KZ)

    # available cfgs:
    # plot_vars: 2D plot
    # c_vars: horizontally-summed vertical chebyshev components
    # f_vars: vertically-summed Fourier components
    # f2_vars: 2D plot w/ horizontal Fourier transform
    # slice_vars: sliced at x=0
    # mean_vars: horizontally averaged
    # sub_vars: 2D plot, mean-subtracted
    def get_plot_vars(cfg):
        ''' unpacks above variables from cfg shorthand '''
        ret_vars = [
            cfg.get('plot_vars', []),
            [i + '_c' for i in cfg.get('c_vars', [])],
            [i + '_f' for i in cfg.get('f_vars', [])],
            cfg.get('f2_vars', []),
            [i + mean_suffix for i in cfg.get('mean_vars', [])],
            [i + slice_suffix for i in cfg.get('slice_vars', [])],
            [i + sub_suffix for i in cfg.get('sub_vars', [])]]
        n_cols = cfg.get('n_cols', sum([len(arr) for arr in ret_vars]))
        n_rows = cfg.get('n_rows', 1)
        ret = [n_cols, n_rows, cfg['save_fmt_str']]
        ret.extend(ret_vars)
        return ret

    plot_cfgs = [
        {
            'save_fmt_str': 'p_%03i.png',
            'mean_vars': ['F_px', 'ux', 'ux_z'],
            'slice_vars': ['uz'],
            'sub_vars': ['ux'],
        },
        {
            'save_fmt_str': 'm_%03i.png',
            'plot_vars': ['ux', 'uz', 'U', 'W'],
        },
        # {
        #     'save_fmt_str': 't_%03i.png',
        #     'mean_vars': ['F_px', 'ux'],
        #     'slice_vars': ['uz'],
        # },
    ]

    dyn_vars = ['uz', 'ux', 'U', 'W', 'ux_z']
    sim_times, domain, state_vars = load(name, params, dyn_vars, plot_stride,
        start=0)

    x = domain.grid(0, scales=params['INTERP_X'])
    z = domain.grid(1, scales=params['INTERP_Z'])
    xmesh, zmesh = quad_mesh(x=x[:, 0], y=z[0])
    x2mesh, z2mesh = quad_mesh(x=np.arange(params['N_X'] // 2), y=z[0])

    # preprocess
    for var in dyn_vars + ['F_px']:
        state_vars[var + mean_suffix] = (
            np.sum(state_vars[var], axis=1) / N_X,
            np.min(state_vars[var], axis=1),
            np.max(state_vars[var], axis=1))

    for var in dyn_vars:
        state_vars[var + slice_suffix] = np.copy(state_vars[var][:, 0, :])

    for var in dyn_vars:
        # can't figure out how to numpy this together
        means = state_vars[var + mean_suffix][0]
        state_vars[var + sub_suffix] = np.copy(state_vars[var])
        for idx, _ in enumerate(state_vars[var + sub_suffix]):
            mean = means[idx]
            state_vars[var + sub_suffix][idx] -= np.tile(mean, (N_X, 1))

    for cfg in plot_cfgs:
        n_cols, n_rows, save_fmt_str, plot_vars, c_vars, f_vars,\
            f2_vars, mean_vars, slice_vars, sub_vars = get_plot_vars(cfg)

        uz_est = params['F'] * get_uz_f_ratio(params)

        for t_idx, sim_time in list(enumerate(sim_times))[rank::size]:
            fig = plt.figure(dpi=200)

            idx = 1
            for var in plot_vars + sub_vars:
                axes = fig.add_subplot(n_rows, n_cols, idx, title=var)

                var_dat = state_vars[var]
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

                var_dat = state_vars[var]
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

            for var in mean_vars + slice_vars:
                axes = fig.add_subplot(n_rows, n_cols, idx, title=var)
                if var in slice_vars:
                    var_dat = state_vars[var]
                else:
                    var_dat, var_min, var_max = state_vars[var]

                z_pts = (zmesh[1:, 0] + zmesh[:-1, 0]) / 2
                p = axes.plot(var_dat[t_idx],
                              z_pts,
                              'r-',
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

                if var == 'F_px%s' % mean_suffix:
                    p = axes.plot(
                        (params['F'] * get_uz_f_ratio(params))**2 / 2
                            * abs(params['KZ'] / params['KX'])
                            * params['RHO0']
                                * np.exp(-params['Z0'] / params['H'])
                            * np.ones(np.shape(z_pts)),
                        z_pts,
                        'orange',
                        linewidth=0.5)

                if var == 'ux%s' % mean_suffix:
                    # mean flow = E[ux * uz] / V_GZ
                    p = axes.plot(
                        uz_est**2 * abs(KZ) / KX / (2 * abs(V_GZ))
                            * np.exp((z_pts - params['Z0']) / (params['H'])),
                        z_pts,
                        'orange',
                        linewidth=0.5)
                    # critical = omega / kx
                    p = axes.plot(
                        params['OMEGA'] / params['KX']
                            * np.ones(np.shape(z_pts)),
                        z_pts,
                        'green',
                        linewidth=0.5)
                if var in mean_vars:
                    p = axes.plot(
                        var_min[t_idx],
                        z_pts,
                        'r:',
                        linewidth=0.2)
                    p = axes.plot(
                        var_max[t_idx],
                        z_pts,
                        'r:',
                        linewidth=0.2)


                plt.xticks(rotation=30)
                plt.yticks(rotation=30)
                xlims = [var_dat.min(), var_dat.max()]
                axes.set_xlim(*xlims)
                axes.set_ylim(z_pts.min(), z_pts.max())
                p = axes.plot(xlims,
                              [params['SPONGE_LOW']
                                  + params['SPONGE_WIDTH']] * len(xlims),
                              'r:',
                              linewidth=0.5)
                p = axes.plot(xlims,
                              [params['SPONGE_HIGH']
                                  - params['SPONGE_WIDTH']] * len(xlims),
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
    ''' few plots for front, defined where flux drops below 1/3 of theory '''
    N_X = params['N_X'] * params['INTERP_X']
    N_Z = params['N_Z'] * params['INTERP_Z']
    dyn_vars = ['uz', 'ux', 'U', 'W', 'ux_z']
    snapshots_dir = SNAPSHOTS_DIR % name
    logfile = '%s/data.log' % snapshots_dir

    sim_times = []
    front_pos = []
    ri_inv = []
    fluxes = []
    flux_th = (params['F'] * get_uz_f_ratio(params))**2 / 2 \
         * abs(params['KZ'] / params['KX']) \
         * params['RHO0'] * np.exp(-params['Z0'] / params['H'])
    flux_threshold = flux_th / 3

    # load if exists
    if not os.path.exists(logfile):
        print('log file not found, generating')
        sim_times, domain, state_vars = load(
            name, params, dyn_vars, 2, start=0)
        x = domain.grid(0, scales=params['INTERP_X'])
        z = domain.grid(1, scales=params['INTERP_Z'])
        xmesh, zmesh = quad_mesh(x=x[:, 0], y=z[0])
        z_pts = (zmesh[1:, 0] + zmesh[:-1, 0]) / 2
        z0 = z[0]

        ux_z = np.sum(state_vars['ux_z'], axis=1) / N_X
        F_px = np.sum(state_vars['F_px'], axis=1) / N_X
        for t_idx, sim_time in enumerate(sim_times):
            max_pos = len(F_px[t_idx]) - 1
            while F_px[t_idx][max_pos] < flux_threshold and max_pos >= 0:
                max_pos -= 1

            front_pos.append(z_pts[max_pos])
            ri_inv.append(ux_z[t_idx][max_pos] / (params['g'] / params['H']))

            fluxes.append(F_px[t_idx][max_pos - 20] -
                          F_px[t_idx][max_pos + 20])
        with open(logfile, 'w') as data:
            data.write(repr(z0.tolist()))
            data.write('\n')
            data.write(repr(sim_times.tolist()))
            data.write('\n')
            data.write(repr(front_pos))
            data.write('\n')
            data.write(repr(ri_inv))
            data.write('\n')
            data.write(repr(fluxes))
            data.write('\n')
            data.write(repr(F_px.tolist()))
            data.write('\n')

    else:
        print('data.log found, loading')
        with open(logfile) as data:
            z0 = np.array(eval(data.readline()))
            sim_times = np.array(eval(data.readline()))
            front_pos = eval(data.readline())
            ri_inv = eval(data.readline())
            fluxes = eval(data.readline())
            F_px = np.array(eval(data.readline()))

    start_idx = 10
    flux_anal = flux_th * np.ones(np.shape(fluxes))
    H = params['H']
    pos_anal = -H * (np.log(sim_times * H / (
        flux_anal / params['RHO0'] * params['KX'] / params['OMEGA'])))
    velocities_anal = np.gradient(pos_anal) / np.gradient(sim_times)
    pos_anal += front_pos[-2] - pos_anal[-2] # fit constant of integration

    tmesh, zmesh = quad_mesh(x=sim_times[start_idx: ], y=z0)
    p = plt.pcolormesh(tmesh,
                       zmesh,
                       F_px[start_idx: , ].T)
    plt.colorbar(p)
    plt.savefig('%s/fpx.png' % snapshots_dir, dpi=200)
    plt.clf()

    plt.plot(sim_times[start_idx: ], front_pos[start_idx: ], label='Data')
    plt.plot(sim_times[start_idx: ], pos_anal[start_idx: ], label='Analytical')
    plt.ylabel('Critical Layer Position')
    plt.xlabel('Time')
    plt.title(name)
    plt.legend()
    plt.savefig('%s/front.png' % snapshots_dir, dpi=200)
    plt.clf()

    plt.plot(sim_times[start_idx: ],
             (np.gradient(front_pos) / np.gradient(sim_times))[start_idx: ],
             label='Data')
    plt.plot(sim_times[start_idx: ], velocities_anal[start_idx: ], label='Analytic')
    plt.ylabel('Critical Layer Velocity')
    plt.xlabel('Time')
    plt.legend()
    plt.title(name)
    plt.savefig('%s/front_v.png' % snapshots_dir, dpi=200)
    plt.clf()

    plt.plot(sim_times[start_idx: ], (1 / np.array(ri_inv[start_idx: ])**2))
    plt.ylabel('Ri')
    plt.xlabel('Time')
    plt.title(name)
    plt.ylim([0, 10])
    plt.savefig('%s/f_ri.png' % snapshots_dir, dpi=200)
    plt.clf()

    plt.plot(sim_times[start_idx: ],
             (np.array(fluxes) * 1e5)[start_idx: ],
             label='Data')
    plt.plot(sim_times[start_idx: ],
             (flux_anal * 1e5)[start_idx: ],
             label='Analytical')
    plt.ylabel('F_px (1e-5)')
    plt.xlabel('Time')
    plt.title(name)
    plt.legend()
    plt.savefig('%s/fluxes.png' % snapshots_dir, dpi=200)
    plt.clf()
