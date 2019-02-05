#!/usr/bin/env python
'''
Absolutely god awful code :(
'''
import logging
import pickle
logger = logging.getLogger()

import os
from collections import defaultdict

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

from dedalus import public as de
from dedalus.tools import post
from dedalus.extras.flow_tools import CFL, GlobalFlowProperty
from dedalus.extras.plot_tools import quad_mesh, pad_limits
from mpi4py import MPI
CW = MPI.COMM_WORLD
PLT_COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'];

SNAPSHOTS_DIR = 'snapshots_%s'
FILENAME_EXPR = '{s}/{s}_s{idx}.h5'
plot_stride = 30

def get_omega(g, h, kx, kz):
    return np.sqrt((g / h) * kx**2 / (kx**2 + kz**2 + 0.25 / h**2))

def get_vgz(g, h, kx, kz):
    return get_omega(g, h, kx, kz) / (kx**2 + kz**2 + 0.25 / h**2) * kz

def get_front_idx(S_px, flux_threshold):
    max_pos = len(S_px) - 1
    while S_px[max_pos] < flux_threshold and max_pos >= 0:
        max_pos -= 1
    return max_pos

def horiz_mean(field, n_x):
    return np.sum(field, axis=1) / n_x

def get_uz_f_ratio(params):
    ''' get uz(z = z0) / F '''
    return (np.sqrt(2 * np.pi) * params['S'] * params['g'] *
            params['KX']**2) * np.exp(-params['S']**2 * params['KZ']**2/2) / (
                2 * params['RHO0'] * np.exp(-params['Z0'] / params['H'])
                * params['OMEGA']**2 * params['KZ'])

def get_flux_th(params):
    return (params['F'] * get_uz_f_ratio(params))**2 / 2 \
        * abs(params['KZ'] / params['KX']) * params['RHO0'] \
        * np.exp(-params['Z0'] / params['H'])

def get_k_damp(params):
    KX = params['KX']
    KZ = params['KZ']
    g = params['g']
    H = params['H']
    k = np.sqrt(KX**2 + KZ**2)

    return params['NU'] * k**5 / abs(KZ * g / H * KX)

def get_anal_uz(params, t, x, z):
    KX = params['KX']
    KZ = params['KZ']
    g = params['g']
    H = params['H']
    Z0 = params['Z0']
    OMEGA = params['OMEGA']
    uz_est = params['F'] * get_uz_f_ratio(params)
    k_damp = get_k_damp(params)

    return uz_est * (
        -np.exp((z - Z0) / 2 * H)
        * np.exp(-k_damp
                 * (z - Z0))
        * np.sin(KX * x + KZ * (z - Z0) - OMEGA * t
                 + 1 / (2 * KZ * H)))

def get_anal_ux(params, t, x, z):
    KX = params['KX']
    KZ = params['KZ']
    g = params['g']
    H = params['H']
    Z0 = params['Z0']
    OMEGA = params['OMEGA']
    ux_est = params['F'] * get_uz_f_ratio(params) * KZ / KX
    k_damp = get_k_damp(params)

    return ux_est * (
        np.exp((z - Z0) / 2 * H)
        * np.exp(-k_damp * (z[0] - Z0))
        * np.sin(KX * x + KZ * (z - Z0) - OMEGA * t
                 + 1 / (KZ * H)))

def get_z_idx(z, z0):
    return int(len(np.where(z0 < z)[0]))

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

def add_nl_problem(problem):
    problem.add_equation('dx(ux) + uz_z = 0')
    problem.add_equation(
        'dt(U) - uz / H' +
        '- NU * (dx(dx(U)) + dz(U_z) - 2 * U_z / H)' +
        '= - sponge * U' +
        '- NL * (ux * dx(U) + uz * dz(U))' +
        '+ NU * (dx(U) * dx(U) + U_z * U_z)' +
        '+ F * exp(-(z - Z0)**2 / (2 * S**2) + Z0 / H) *' +
            'cos(KX * x - OMEGA * t)')
    problem.add_equation(
        'dt(ux) + dx(W) + (g * H) * dx(U)' +
        '- (NU * dx(dx(ux)) + NU * dz(ux_z))' +
        '+ NU * dz(ux) / H'
        '= - sponge * ux' +
        '- NL * (ux * dx(ux) + uz * dz(ux))' +
        '- NU * (dx(U) * dx(ux) + U_z * ux_z)' +
        '+ NU * ux * (dx(dx(U)) + dz(U_z))' +
        '+ NU * ux * (dx(U) * dx(U) + U_z * U_z)' +
        '- 2 * NU * ux * U_z / H' +
        # '+ NU * ux * (1 - exp(-U)) / H**2' +
        '+ NU * ux * U / H**2' +
        '- NL * (W * dx(U))')
    problem.add_equation(
        'dt(uz) + dz(W) + (g * H) * dz(U) - W/H' +
        '- (NU * dx(dx(uz)) + NU * dz(uz_z))' +
        '+ NU * dz(uz) / H'
        '= - sponge * uz - NL * (ux * dx(uz) + uz * dz(uz))' +
        '- NU * (dx(U) * dx(uz) + U_z * uz_z)' +
        '+ NU * uz * (dx(dx(U)) + dz(U_z))' +
        '+ NU * uz * (dx(U) * dx(U) + U_z * U_z)' +
        '- 2 * NU * uz * U_z / H' +
        # '+ NU * uz * (1 - exp(-U)) / H**2' +
        '+ NU * uz * U / H**2' +
        '- NL * (W * dz(U))')
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

def add_lin_problem(problem):
    problem.add_equation('dx(ux) + dz(uz) = 0')
    problem.add_equation(
        'dt(U) - uz / H' +
        '= - sponge * U' +
        '+ F * exp(-(z - Z0)**2 / (2 * S**2) + Z0 / H) *' +
            'cos(KX * x - OMEGA * t)')
    problem.add_equation(
        'dt(ux) + dx(W) + (g * H) * dx(U)' +
        '= - sponge * ux')
    problem.add_equation(
        'dt(uz) + dz(W) + (g * H) * dz(U) - W/H' +
        '= - sponge * uz')
    # don't really need, but too messy to refactor everywhere else
    # problem.add_equation('dz(ux) - ux_z = 0')

    problem.add_bc('right(uz) = 0')
    problem.add_bc('left(W) = 0', condition='nx == 0')
    problem.add_bc('left(uz) = 0', condition='nx != 0')

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
    NL = params['NL']

    variables = ['W', 'U', 'ux', 'uz']
    if NL:
        variables += ['uz_z', 'ux_z', 'U_z']
    problem = de.IVP(domain, variables=variables)
    problem.parameters.update(params)

    problem.substitutions['sponge'] = 'SPONGE_STRENGTH * 0.5 * ' +\
        '(2 + tanh((z - SPONGE_HIGH) / (SPONGE_WIDTH * (ZMAX - SPONGE_HIGH)))'+\
        '- tanh((z - SPONGE_LOW) / (SPONGE_WIDTH * (SPONGE_LOW))))'
    problem.substitutions['rho0'] = 'RHO0 * exp(-z / H)'
    if NL:
        add_nl_problem(problem)
    else:
        add_lin_problem(problem)

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
              min_dt=0.007,
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
    flow.add_property('sqrt(ux**2 + ux**2)', name='u')

    # Main loop
    logger.info('Starting sim...')
    while solver.ok:
        cfl_dt = cfl.compute_dt() if params['NL'] else params['DT']
        solver.step(cfl_dt)
        curr_iter = solver.iteration

        if curr_iter % int((params['T_F'] / params['DT']) /
                           params['NUM_SNAPSHOTS']) == 0:
            logger.info('Reached time %f out of %f, timestep %f vs max %f',
                        solver.sim_time,
                        solver.stop_sim_time,
                        cfl_dt,
                        params['DT'])
            logger.info('Max u = %e' % flow.max('u'))

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
                state_vars['%s_c' % varname].append(
                    np.copy(np.abs(values['c'])))

        total_sim_times.extend(sim_times)
        i += 1
        filename = FILENAME_EXPR.format(s=snapshots_dir, idx=i)

    # cast to np arrays
    for key in state_vars.keys():
        state_vars[key] = np.array(state_vars[key])

    if not params['NL']:
        state_vars['ux_z'] = np.gradient(state_vars['ux'], axis=2)

    state_vars['S_{px}'] = params['RHO0'] * np.exp(-z/ params['H']) * (
        (state_vars['ux'] * state_vars['uz']) +
        (params['NU'] * np.exp(state_vars['U']) * state_vars['ux_z']))
    return np.array(total_sim_times[start::plot_stride]), domain, state_vars

def plot(name, params):
    rank = CW.rank
    size = CW.size

    slice_suffix = '(x=0)'
    mean_suffix = '(mean)'
    sub_suffix = ' (- mean)'
    res_suffix = ' (res)'
    snapshots_dir = SNAPSHOTS_DIR % name
    matplotlib.rcParams.update({'font.size': 6})
    N_X = params['N_X'] * params['INTERP_X']
    N_Z = params['N_Z'] * params['INTERP_Z']
    KX = params['KX']
    KZ = params['KZ']
    g = params['g']
    H = params['H']
    Z0 = params['Z0']
    OMEGA = params['OMEGA']
    V_GZ = get_vgz(g, H, KX, KZ)

    # available cfgs:
    # plot_vars: 2D plot
    # c_vars: horizontally-summed vertical chebyshev components
    # f_vars: vertically-summed Fourier components
    # f2_vars: 2D plot w/ horizontal Fourier transform
    # slice_vars: sliced at x=0
    # mean_vars: horizontally averaged
    # sub_vars: 2D plot, mean-subtracted
    # res_vars: subtract analytical solution (only uz/ux)
    def get_plot_vars(cfg):
        ''' unpacks above variables from cfg shorthand '''
        ret_vars = [
            cfg.get('plot_vars', []),
            [i + '_c' for i in cfg.get('c_vars', [])],
            [i + '_f' for i in cfg.get('f_vars', [])],
            cfg.get('f2_vars', []),
            [i + mean_suffix for i in cfg.get('mean_vars', [])],
            [i + slice_suffix for i in cfg.get('slice_vars', [])],
            [i + sub_suffix for i in cfg.get('sub_vars', [])],
            [i + res_suffix for i in cfg.get('res_vars', [])]]
        n_cols = cfg.get('n_cols', sum([len(arr) for arr in ret_vars]))
        n_rows = cfg.get('n_rows', 1)
        ret = [n_cols, n_rows, cfg['save_fmt_str']]
        ret.extend(ret_vars)
        return ret

    plot_cfgs = [
        {
            'save_fmt_str': 'p_%03i.png',
            'plot_vars': ['ux', 'uz'],
            'res_vars': ['ux', 'uz'],
        },
        # {
        #     'save_fmt_str': 's_%03i.png',
        #     'slice_vars': ['uz', 'ux'],
        #     'mean_vars': ['S_{px}', 'ux'],
        # },
        # {
        #     'save_fmt_str': 'm_%03i.png',
        #     'plot_vars': ['uz', 'ux', 'W', 'U'],
        # },
    ]

    dyn_vars = ['uz', 'ux', 'U', 'W']
    if params['NL']:
        dyn_vars += ['ux_z']
    sim_times, domain, state_vars = load(name, params, dyn_vars, plot_stride,
        start=0)

    x = domain.grid(0, scales=params['INTERP_X'])
    z = domain.grid(1, scales=params['INTERP_Z'])
    xmesh, zmesh = quad_mesh(x=x[:, 0], y=z[0])
    x2mesh, z2mesh = quad_mesh(x=np.arange(N_X // 2), y=z[0])

    # preprocess
    for var in dyn_vars + ['S_{px}']:
        state_vars[var + mean_suffix] = (
            horiz_mean(state_vars[var], N_X),
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
            f2_vars, mean_vars, slice_vars, sub_vars, res_vars\
            = get_plot_vars(cfg)

        uz_est = params['F'] * get_uz_f_ratio(params)
        ux_est = uz_est * KZ / KX
        uz_mean = np.outer(np.ones(N_X),
                           state_vars['uz%s' % mean_suffix][0][t_idx])
        ux_mean = np.outer(np.ones(N_X),
                           state_vars['ux%s' % mean_suffix][0][t_idx])

        for t_idx, sim_time in list(enumerate(sim_times))[rank::size]:
            fig = plt.figure(dpi=400)

            uz_anal = get_anal_uz(params, sim_time, x, z)
            ux_anal = get_anal_ux(params, sim_time, x, z)

            idx = 1
            for var in plot_vars + sub_vars + res_vars:
                if res_suffix in var:
                    # divide by analytical profile and normalize
                    if var == 'uz%s' % res_suffix:
                        var_dat = (state_vars['uz'][t_idx] - uz_anal - uz_mean)\
                            / (uz_est * np.exp((z - Z0) / (2 * H)))
                        title = 'u_z'
                    elif var == 'ux%s' % res_suffix:
                        var_dat = (state_vars['ux'][t_idx] - ux_anal - ux_mean)\
                            / (ux_est * np.exp((z - Z0) / (2 * H)))
                        title = 'u_x'
                    else:
                        raise ValueError('lol wtf is %s' % var)
                    z_bot = len(np.where(z[0] < Z0 + 3 * params['S'])[0])

                    S_px_mean = state_vars['S_{px}%s' % mean_suffix][0]
                    front_idx = get_front_idx(S_px_mean[t_idx],
                                              get_flux_th(params) * 0.2)
                    z_top = len(np.where(z[0] < params['SPONGE_HIGH']
                                         - 2 * params['SPONGE_WIDTH'])[0])
                    z_top = front_idx
                    # truncate uz_anal at sponge and critical layers
                    var_dat[:, 0: z_bot] = 0
                    var_dat[:, z_top: ] = 0

                    err = np.sqrt(np.mean(var_dat**2))
                    axes = fig.add_subplot(
                        n_rows,
                        n_cols,
                        idx,
                        title=r'$\delta %s$ (RMS = %.4e)' % (title, err))
                    vmin = var_dat.min()
                    vmax = var_dat.max()
                else:
                    axes = fig.add_subplot(
                        n_rows,
                        n_cols,
                        idx,
                        title=r'$%s$' % var)

                    var_dat = state_vars[var][t_idx]
                    vmin = state_vars[var].min()
                    vmax = state_vars[var].max()
                p = axes.pcolormesh(xmesh,
                                    zmesh,
                                    var_dat.T,
                                    vmin=vmin, vmax=vmax)
                axes.axis(pad_limits(xmesh, zmesh))
                cb = fig.colorbar(p, ax=axes)
                plt.xticks(rotation=30)
                plt.yticks(rotation=30)
                idx += 1

            for var in f2_vars:
                axes = fig.add_subplot(n_rows, n_cols, idx,
                                       title=r'$\log %s$ (x-FT)' % var)

                var_dat = state_vars[var]
                var_dat_t = np.fft.fft(var_dat[t_idx], axis=0)
                var_dat_shaped = np.log(np.abs(
                    2 * var_dat_t.real[:N_X // 2, :]))
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
                axes = fig.add_subplot(n_rows, n_cols, idx, title=r'$%s$' % var)
                if var in slice_vars:
                    var_dat = state_vars[var]
                else:
                    var_dat, var_min, var_max = state_vars[var]

                p = axes.plot(var_dat[t_idx],
                              z[0],
                              'r-',
                              linewidth=0.5)
                if var == 'uz%s' % slice_suffix:
                    p = axes.plot(
                        uz_anal[0, :],
                        z[0],
                        'orange',
                        linewidth=0.5)
                    p = axes.plot(
                        -uz_est * np.exp((z[0] - Z0) / (2 * H)), z[0], 'g',
                        uz_est * np.exp((z[0] - Z0) / (2 * H)), z[0], 'g',
                        linewidth=0.5)

                if var == 'ux%s' % slice_suffix:
                    p = axes.plot(
                        ux_anal[0, :],
                        z[0],
                        'orange',
                        linewidth=0.5)
                    p = axes.plot(
                        -ux_est * np.exp((z[0] - Z0) / (2 * H)), z[0], 'g',
                        ux_est * np.exp((z[0] - Z0) / (2 * H)), z[0], 'g',
                        linewidth=0.5)

                if var == 'S_{px}%s' % mean_suffix:
                    p = axes.plot(
                        uz_est**2 / 2
                            * abs(KZ / KX)
                            * params['RHO0'] * np.exp(-params['Z0'] / H)
                            * np.ones(np.shape(z[0])),
                        z[0],
                        'orange',
                        linewidth=0.5)

                if var == 'ux%s' % mean_suffix:
                    # mean flow = E[ux * uz] / V_GZ
                    p = axes.plot(
                        uz_est**2 * abs(KZ) / KX / (2 * abs(V_GZ))
                            * np.exp((z[0] - Z0) / H),
                        z[0],
                        'orange',
                        linewidth=0.5)
                    # critical = omega / kx
                    p = axes.plot(OMEGA / KX * np.ones(np.shape(z[0])),
                        z[0],
                        'green',
                        linewidth=0.5)
                if var in mean_vars:
                    p = axes.plot(
                        var_min[t_idx],
                        z[0],
                        'r:',
                        linewidth=0.2)
                    p = axes.plot(
                        var_max[t_idx],
                        z[0],
                        'r:',
                        linewidth=0.2)


                plt.xticks(rotation=30)
                plt.yticks(rotation=30)
                xlims = [var_dat.min(), var_dat.max()]
                axes.set_xlim(*xlims)
                axes.set_ylim(z[0].min(), z[0].max())
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
                              [Z0 + 3 * params['S']] * len(xlims),
                              'b--',
                              linewidth=0.5)
                p = axes.plot(xlims,
                              [Z0 - 3 * params['S']] * len(xlims),
                              'b--',
                              linewidth=0.5)
                idx += 1
            for var in c_vars:
                axes = fig.add_subplot(n_rows,
                                       n_cols,
                                       idx,
                                       title=r'$%s$ (kx=kx_d)' % var)
                var_dat = state_vars[var]
                kx_idx = round(KX / (2 * np.pi / params['XMAX']))
                p = axes.semilogx(var_dat[t_idx][kx_idx],
                                  range(len(var_dat[t_idx][kx_idx])),
                                  linewidth=0.5)
                idx += 1

            for var in f_vars:
                axes = fig.add_subplot(n_rows,
                                       n_cols,
                                       idx,
                                       title=r'$%s$ (Cheb. summed)' % var)
                var_dat = state_vars[var.replace('_f', '_c')]
                summed_dat = np.sum(np.abs(var_dat[t_idx]), 1)
                p = axes.semilogx(summed_dat,
                                  range(len(summed_dat)),
                                  linewidth=0.5)
                idx += 1

            fig.suptitle(
                r'Config: $%s (t=%.2f, \vec{k}=(%.2f, %.2f), \omega=%.2f)$' %
                (name, sim_time, KX, KZ, OMEGA))
            fig.subplots_adjust(hspace=0.7, wspace=0.6)
            savefig = save_fmt_str % (t_idx)
            plt.savefig('%s/%s' % (snapshots_dir, savefig))
            logger.info('Saved %s/%s' % (snapshots_dir, savefig))
            plt.close()

def write_front(name, params):
    ''' few plots for front, defined where flux drops below 1/2 of theory '''
    N_X = params['N_X'] * params['INTERP_X']
    N_Z = params['N_Z'] * params['INTERP_Z']
    H = params['H']
    KX = params['KX']
    KZ = params['KZ']
    OMEGA = params['OMEGA']
    u_c = OMEGA / KX
    dyn_vars = ['uz', 'ux', 'U', 'W']
    if params['NL']:
        dyn_vars += ['ux_z']
    snapshots_dir = SNAPSHOTS_DIR % name
    logfile = '%s/data.pkl' % snapshots_dir

    flux_th = get_flux_th(params)
    flux_threshold = flux_th * 0.3

    # load if exists
    if not os.path.exists(logfile):
        print('log file not found, generating')
        sim_times, domain, state_vars = load(
            name, params, dyn_vars, plot_stride=1, start=0)
        x = domain.grid(0, scales=params['INTERP_X'])
        z = domain.grid(1, scales=params['INTERP_Z'])
        xmesh, zmesh = quad_mesh(x=x[:, 0], y=z[0])
        z0 = z[0]

        ux_res_slice = []
        uz_res_slice = []

        S_px = horiz_mean(state_vars['S_{px}'], N_X)
        u0 = horiz_mean(state_vars['ux'], N_X)

        uz_est = params['F'] * get_uz_f_ratio(params)
        ux_est = uz_est * KZ / KX
        for t_idx, sim_time in enumerate(sim_times):
            ux_res = (get_anal_ux(params, sim_time, x, z)
                      - state_vars['ux'][t_idx]) / \
                (ux_est * np.exp((z - params['Z0']) / (2 * H)))
            uz_res = (get_anal_uz(params, sim_time, x, z)
                      - state_vars['uz'][t_idx]) / \
                (uz_est * np.exp((z - params['Z0']) / (2 * H)))
            front_idx = get_front_idx(S_px[t_idx], flux_threshold)
            slice_idx = get_z_idx(z0[front_idx] - 1 / KZ, z0)

            ux_res_slice.append(ux_res[:, slice_idx - 1])
            uz_res_slice.append(uz_res[:, slice_idx - 1])

        with open(logfile, 'wb') as data:
            pickle.dump((z0, sim_times, S_px, u0, ux_res_slice, uz_res_slice),
                        data)
    else:
        print('log file found, not regenerating')

def plot_front(name, params):
    N_X = params['N_X'] * params['INTERP_X']
    N_Z = params['N_Z'] * params['INTERP_Z']
    H = params['H']
    KX = params['KX']
    KZ = params['KZ']
    OMEGA = params['OMEGA']
    u_c = OMEGA / KX
    flux_th = get_flux_th(params)
    flux_threshold = flux_th * 0.3
    k_damp = get_k_damp(params)
    start_idx = 10

    dyn_vars = ['uz', 'ux', 'U', 'W']
    snapshots_dir = SNAPSHOTS_DIR % name
    logfile = '%s/data.pkl' % snapshots_dir
    if params['NL']:
        dyn_vars += ['ux_z']
    if not os.path.exists(logfile):
        write_front(name, params)

    print('loading data')
    with open(logfile, 'rb') as data:
        z0, sim_times, S_px, u0, ux_res_slice, uz_res_slice\
            = pickle.load(data)

    tf = sim_times[-1]
    t = sim_times[start_idx: ]
    dt = np.mean(np.gradient(sim_times[1: -1])) # should be T_F / NUM_SNAPSHOTS

    S_px0 = [] # S_px averaged near origin
    dSpx_S = [] # Delta S_px using S criterion
    dSpx_U = [] # Delta S_px using U criterion
    front_pos_S = [] # front position using S criterion
    front_pos_U = [] # front position using U criterion
    dz = abs(1 / params['KZ'])
    l_z = abs(2 * np.pi / params['KZ'])
    z_b = params['Z0'] + 3 * params['S']
    z_b_idx = get_z_idx(z_b, z0)
    for t_idx, sim_time in enumerate(sim_times):
        front_idx_S = get_front_idx(S_px[t_idx], flux_threshold)
        z_cS = z0[front_idx_S]
        front_pos_S.append(z_cS)

        front_idx_U = 0
        u_curr = u0[t_idx]
        while front_idx_U < len(u_curr) - 1 and u_curr[front_idx_U] < u_c:
            front_idx_U += 1
        z_cU = z0[front_idx_U]
        front_pos_U.append(z_cU)

        # measure flux incident at dz critical layer
        dz_idx = get_z_idx(dz, z0)
        S_px0.append(-np.mean(S_px[
            t_idx, get_z_idx(z_b, z0): get_z_idx(z_b + l_z, z0)]))
        dSpx_S.append(-np.mean(S_px[
            t_idx, get_z_idx(z_cS - l_z - dz, z0): get_z_idx(z_cS - dz, z0)]))
        dSpx_U.append(-np.mean(S_px[
            t_idx, get_z_idx(z_cU - l_z - dz, z0): get_z_idx(z_cU - dz, z0)]))

    times = [int((len(sim_times) - start_idx) * time_frac + start_idx - 1)
             for time_frac in [1/8, 3/8, 5/8, 7/8]]
    fig = plt.figure()
    if 'lin' in name:
        #####################################################################
        # fluxes.png
        #
        # horizontal plot showing Fpx at certain times
        #####################################################################
        z0_cut = z0[z_b_idx: ]
        for time in times:
            plt.plot(z0_cut,
                     S_px[time, z_b_idx: ] / flux_th,
                     linewidth=0.7,
                     label=r't=%.1f$N^{-1}$' % sim_times[time])
        plt.plot(z0_cut,
                 np.exp(-k_damp * 2 * (z0_cut - params['Z0'])),
                 linewidth=1.5,
                 label=r'Model')
        plt.xlim(z_b, params['ZMAX'])
        plt.ylim(-0.1, 1.1 * S_px[time, z_b_idx: ].max() / flux_th)
        plt.legend(fontsize=6)

        plt.xlabel(r'$z(H)$')
        plt.ylabel(r'$S_{px} / S_0$')
        plt.savefig('%s/fluxes.png' % snapshots_dir, dpi=400)
        plt.close()

    else:
        #####################################################################
        # fluxes.png
        #
        # plot fluxes + mean flow over time
        #####################################################################
        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        f.subplots_adjust(hspace=0)
        z0_cut = z0[z_b_idx: ]
        for time, color in zip(times, PLT_COLORS):
            S_px_avg = np.sum(S_px[time - 2: time + 2, z_b_idx: ], axis=0) / 4
            # mean flow
            ax1.plot(z0_cut,
                     u0[time, z_b_idx: ] / u_c,
                     '%s-' % color,
                     linewidth=0.7,
                     label=r't=%.1f$N^{-1}$' % sim_times[time])
            # plot S_px sliced at time
            ax2.plot(z0_cut,
                     S_px[time, z_b_idx: ] / flux_th,
                     '%s-' % color,
                     linewidth=0.7,
                     label=r't=%.1f$N^{-1}$' % sim_times[time])
            # plot S_px averaged in ~ 1 period
            ax2.plot(z0_cut,
                     S_px_avg / flux_th,
                     '%s:' % color,
                     linewidth=0.7,
                     label=r't=%.1f$N^{-1}$' % sim_times[time])
        # overlay analytical flux including viscous dissipation
        ax2.plot(z0_cut,
                 np.exp(-k_damp * 2 * (z0_cut - params['Z0'])),
                 linewidth=1.5,
                 label=r'$\nu$-only')
        # indicate vertical averaging wavelength
        ax2.axvspan(front_pos_S[times[-1]] - dz - l_z,
                    front_pos_S[times[-1]] - dz,
                    color='grey')
        ax1.set_xlim(z_b, params['ZMAX'])
        ax1.set_ylim(-0.1, 1.1 * u0.max() / u_c)
        ax2.set_xlim(z_b, params['ZMAX'])
        ax2.set_ylim(-0.1, 1.1 * S_px.max() / flux_th)
        ax2.legend(fontsize=6)

        ax1.set_ylabel(r'$U_0 / c_{ph, x}$')
        ax2.set_ylabel(r'$S_{px} / S_0$')
        ax2.set_xlabel(r'$z(H)$')
        plt.savefig('%s/fluxes.png' % snapshots_dir, dpi=400)
        plt.close()

        #####################################################################
        # front.png
        #
        # plot front position and absorbed flux over time
        #####################################################################
        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        f.subplots_adjust(hspace=0)

        zf = np.max(front_pos_S[-5: -1])

        S_px0 = np.array(S_px0)
        dSpx_S = np.array(dSpx_S)
        dSpx_U = np.array(dSpx_U)
        front_pos_S = np.array(front_pos_S)
        front_pos_U = np.array(front_pos_U)

        # compute front position
        front_vel_S = dSpx_S / (params['RHO0'] * np.exp(-front_pos_S / H) * u_c)
        front_pos_intg_S = np.cumsum(front_vel_S[start_idx: ]) * dt
        front_pos_intg_S += zf - front_pos_intg_S[-1]
        front_vel_U = dSpx_U / (params['RHO0'] * np.exp(-front_pos_U / H) * u_c)
        front_pos_intg_U = np.cumsum(front_vel_U[start_idx: ]) * dt
        front_pos_intg_U += zf - front_pos_intg_U[-1]

        # estimate incident Delta S_px if all from S_px0 that's viscously
        # damped, compare to other Delta S_px criteria/from data
        color_idx = 0
        dSpx0 = -S_px0[start_idx: ] / flux_th * \
            np.exp(-k_damp * 2 * (front_pos_S[start_idx: ] - (z_b + l_z / 2)))
        ax1.plot(t,
                 dSpx0,
                 '%s-' % PLT_COLORS[color_idx],
                 label=r'$\Delta S_{px,0}|_{z=z_{c;S}}$',
                 linewidth=0.7)
        color_idx += 1
        ax1.plot(t,
                 -dSpx_S[start_idx: ] / flux_th,
                 '%s-' % PLT_COLORS[color_idx],
                 label=r'$\Delta S_{px}(z_{c; S})$',
                 linewidth=0.7)
        color_idx += 1
        ax1.plot(t,
                 -dSpx_U[start_idx: ] / flux_th,
                 '%s-' % PLT_COLORS[color_idx],
                 label=r'$\Delta S_{px}(z_{c; U})$',
                 linewidth=0.7)
        ax1.set_ylabel(r'$S_{px} / S_{px, 0}$')
        ax1.legend(fontsize=6)

        # compare forecasts of front position using three predictors integrated
        # from incident flux in data
        color_idx = 0
        ax2.plot(t,
                 front_pos_intg_S,
                 '%s-' % PLT_COLORS[color_idx],
                 label='Model (data $\Delta S_{px}(z_{c; S})$)',
                 linewidth=0.7)
        color_idx += 1
        ax2.plot(t,
                 front_pos_S[start_idx: ],
                 '%s-' % PLT_COLORS[color_idx],
                 label='Data (S)',
                 linewidth=0.7)
        color_idx += 1
        ax2.plot(t,
                 front_pos_U[start_idx: ],
                 '%s-' % PLT_COLORS[color_idx],
                 label='Data (U)',
                 linewidth=0.7)
        mean_pos = np.max(front_pos_S[len(front_pos_S) // 3: ])
        color_idx += 1

        # three multipliers are (i) average incident flux, (ii) estimated
        # incident flux extrapolated from nu and (iii) full flux
        mean_incident = np.mean(-dSpx_S[len(dSpx_S) // 3: ])
        est_generated_flux = -S_px0[len(S_px0) // 5]
        est_incident_flux = est_generated_flux *\
            np.exp(-k_damp * 2 * (mean_pos - params['Z0']))
        flux_mults = [mean_incident / flux_th,
                      est_incident_flux / flux_th,
                      1]
        for mult in flux_mults:
            tau = H * params['RHO0'] * u_c / (flux_th * mult)
            pos_anal = -H * np.log(
                (sim_times - tf + tau * np.exp(-zf/params['H']))
                / tau)
            ax2.plot(t,
                     pos_anal[start_idx: ],
                     '%s:' % PLT_COLORS[color_idx],
                     label='Model ($%.2f S_{px,0}$)' % mult,
                     linewidth=0.7)
            color_idx += 1
        ax2.set_ylabel(r'$z_c$')
        ax2.set_xlabel(r't')
        ax2.set_ylim([zf, max(front_pos_S) * 1.05])
        ax2.legend(fontsize=6)
        plt.savefig('%s/front.png' % snapshots_dir, dpi=400)
        plt.close()

    #########################################################################
    # fft.png
    #
    # plot FFTs of residuals
    #########################################################################
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    f.subplots_adjust(hspace=0)
    num_plots = 5
    half_avg = 8
    ux_ffts = np.array(
        [np.abs(np.fft.fft(i) / N_X)[1: N_X//2] for i in ux_res_slice])
    uz_ffts = np.array(
        [np.abs(np.fft.fft(i) / N_X)[1: N_X//2] for i in uz_res_slice])
    kx = np.linspace(0, 2 * np.pi * (N_X // 2) / params['XMAX'],
                     N_X // 2)[1: ]
    # drop endpoints
    for idx in np.linspace(0, len(ux_ffts), num_plots + 2)[1: -1]:
        idx = int(idx)
        x_avg = np.mean(ux_ffts[idx - half_avg:idx + half_avg, :], 0)
        z_avg = np.mean(uz_ffts[idx - half_avg:idx + half_avg, :], 0)
        ax1.loglog(kx, x_avg, label='t=%.1f' % sim_times[idx], linewidth=0.7)
        ax2.loglog(kx, z_avg, label='t=%.1f' % sim_times[idx], linewidth=0.7)
    visc_kx = np.sqrt(OMEGA / (2 * params['NU']))
    ax1.axvline(x=visc_kx, linewidth=1.5, color='red')
    ax2.axvline(x=visc_kx, linewidth=1.5, color='red')
    ax1.set_ylabel(r'$\tilde{u}_x(k_x)$')
    ax2.set_ylabel(r'$\tilde{u}_z(k_x)$')
    ax2.set_xlabel(r'$k_x$')
    ax1.legend()
    ax2.legend()
    plt.savefig('%s/fft.png' % snapshots_dir, dpi=400)
    plt.close()
