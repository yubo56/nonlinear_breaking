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
from scipy.interpolate import interp1d

from dedalus import public as de
from dedalus.tools import post
from dedalus.extras.flow_tools import CFL, GlobalFlowProperty
from dedalus.extras.plot_tools import quad_mesh, pad_limits
from mpi4py import MPI
CW = MPI.COMM_WORLD
PLT_COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'];

SNAPSHOTS_DIR = 'snapshots_%s'
FILENAME_EXPR = '{s}/{s}_s{idx}.h5'
plot_stride = 10

def populate_globals(var_dict):
    for key, val in var_dict.items():
        exec(key + "=" + repr(val), globals())

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
    populate_globals(params)
    return (np.sqrt(2 * np.pi) * S * g *
            KX**2) * np.exp(-S**2 * KZ**2/2) / (
                2 * RHO0 * np.exp(-Z0 / H)
                * OMEGA**2 * KZ)

def get_flux_th(params):
    return (F * get_uz_f_ratio(params))**2 / 2 \
        * abs(KZ / KX) * RHO0 * np.exp(-Z0 / H)

def get_k_damp(params):
    populate_globals(params)
    k = np.sqrt(KX**2 + KZ**2)

    return NU * k**5 / abs(KZ * g / H * KX)

def get_anal_uz(params, t, x, z):
    populate_globals(params)
    uz_est = F * get_uz_f_ratio(params)
    k_damp = get_k_damp(params)

    return uz_est * (
        -np.exp((z - Z0) / 2 * H)
        * np.exp(-k_damp * (z - Z0))
        * np.sin(KX * x + KZ * (z - Z0) - OMEGA * t
                 + 1 / (2 * KZ * H)))

def get_anal_ux(params, t, x, z):
    populate_globals(params)
    uz_est = F * get_uz_f_ratio(params)
    k_damp = get_k_damp(params)

    return uz_est * np.exp((z - Z0) / 2 * H) * np.exp(-k_damp * (z[0] - Z0)) * (
        KZ / KX * np.sin(KX * x + KZ * (z - Z0) - OMEGA * t + 1 / (2 * KZ * H))
        - np.cos(KX * x + KZ * (z - Z0) - OMEGA * t + 1 / (2 * KZ * H))
            / (2 * H * KX)
    )

def get_idx(z, z0):
    return int(len(np.where(z0 < z)[0]))

def get_times(time_fracs, sim_times, start_idx):
    return [int((len(sim_times) - start_idx) * time_frac + start_idx - 1)
            for time_frac in time_fracs]

def set_ic(name, solver, domain, params):
    populate_globals(params)
    snapshots_dir = SNAPSHOTS_DIR % name
    filename = FILENAME_EXPR.format(s=snapshots_dir, idx=1)

    if os.path.exists(snapshots_dir):
        # snapshots exist, merge if need and then load
        print('Attempting to load snapshots')
        write, dt = solver.load_state(filename, -1)
        print('Loaded snapshots')
        return write, dt
    print('No snapshots found')
    return 0, DT


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

    problem.add_bc('right(uz) = 0')
    problem.add_bc('left(W) = 0', condition='nx == 0')
    problem.add_bc('left(uz) = 0', condition='nx != 0')

def add_nl_problem(problem):
    problem.add_equation('dx(ux) + uz_z = 0')
    problem.add_equation(
        'dt(U) - uz / H' +
        '- NU * (dx(dx(U)) + dz(U_z))'
        '= - sponge * U' +
        '+ F * exp(-(z - Z0)**2 / (2 * S**2) + Z0 / H) *' +
            'cos(KX * x - OMEGA * t)' +
        '+ NL_MASK * (- (ux * dx(U) + uz * dz(U))' +
        '- NU * 2 * U_z / H' +
        '+ NU * (dx(U) * dx(U) + U_z * U_z))')
    problem.add_equation(
        'dt(ux) + dx(W) + (g * H) * dx(U)' +
        '- NU * (dx(dx(ux)) + dz(ux_z))' +
        '= - sponge * ux' +
        '+ NL_MASK * (- (ux * dx(ux) + uz * dz(ux))'
        '- NU * dz(ux) / H' +
        '- NU * (dx(U) * dx(ux) + U_z * ux_z)' +
        '+ NU * ux * (dx(dx(U)) + dz(U_z))' +
        '+ NU * ux * (dx(U) * dx(U) + U_z * U_z)' +
        '- 2 * NU * ux * U_z / H' +
        '+ NU * ux * (1 - exp(-U)) / H**2' +
        '- (W * dx(U)))')
    problem.add_equation(
        'dt(uz) + dz(W) + (g * H) * U_z - W/H' +
        '- NU * (dx(dx(uz)) + dz(uz_z))' +
        '= - sponge * uz' +
        '+ NL_MASK * (- (ux * dx(uz) + uz * dz(uz))' +
        '- NU * dz(uz) / H'
        '- NU * (dx(U) * dx(uz) + U_z * uz_z)' +
        '+ NU * uz * (dx(dx(U)) + dz(U_z))' +
        '+ NU * uz * (dx(U) * dx(U) + U_z * U_z)' +
        '- 2 * NU * uz * U_z / H' +
        '+ NU * uz * (1 - exp(-U)) / H**2' +
        '- (W * dz(U)))')
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

def get_solver(params):
    ''' sets up solver '''
    populate_globals(params)
    x_basis = de.Fourier('x',
                         N_X // INTERP_X,
                         interval=(0, XMAX),
                         dealias=3/2)
    z_basis = de.Chebyshev('z',
                           N_Z // INTERP_Z,
                           interval=(0, ZMAX),
                           dealias=3/2)
    domain = de.Domain([x_basis, z_basis], np.float64)
    z = domain.grid(1)

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
        if mask:
            problem.substitutions['NL_MASK'] = \
                '0.5 * (1 + tanh((z - (Z0 + 4 * S)) / S))'
        else:
            problem.substitutions['NL_MASK'] = '1'
        add_nl_problem(problem)
    else:
        add_lin_problem(problem)

    # Build solver
    solver = problem.build_solver(de.timesteppers.RK443)
    solver.stop_sim_time = T_F
    solver.stop_wall_time = np.inf
    solver.stop_iteration = np.inf
    return solver, domain

def run_strat_sim(set_ICs, name, params):
    populate_globals(params)
    snapshots_dir = SNAPSHOTS_DIR % name

    solver, domain = get_solver(params)

    # Initial conditions
    _, dt = set_ICs(name, solver, domain, params)

    cfl = CFL(solver,
              initial_dt=dt,
              cadence=5,
              max_dt=DT,
              min_dt=0.007,
              safety=0.5,
              threshold=0.10)
    cfl.add_velocities(('ux', 'uz'))
    cfl.add_frequency(DT)
    snapshots = solver.evaluator.add_file_handler(
        snapshots_dir,
        sim_dt=T_F / NUM_SNAPSHOTS)
    snapshots.add_system(solver.state)

    # Flow properties
    flow = GlobalFlowProperty(solver, cadence=10)
    flow.add_property('sqrt(ux**2 + ux**2)', name='u')

    # Main loop
    logger.info('Starting sim...')
    while solver.ok:
        cfl_dt = cfl.compute_dt() if NL else DT
        solver.step(cfl_dt)
        curr_iter = solver.iteration

        if curr_iter % int((T_F / DT) /
                           NUM_SNAPSHOTS) == 0:
            logger.info('Reached time %f out of %f, timestep %f vs max %f',
                        solver.sim_time,
                        solver.stop_sim_time,
                        cfl_dt,
                        DT)
            logger.info('Max u = %e' % flow.max('u'))

def merge(name):
    snapshots_dir = SNAPSHOTS_DIR % name
    dir_expr = '{s}/{s}_s{idx}'

    idx = 1
    to_merge = False

    snapshots_piece_dir = dir_expr.format(s=snapshots_dir, idx=idx)
    filename = FILENAME_EXPR.format(s=snapshots_dir, idx=idx)

    while os.path.exists(snapshots_piece_dir):
        if not os.path.exists(filename):
            to_merge = True
        idx += 1
        snapshots_piece_dir = dir_expr.format(s=snapshots_dir, idx=idx)
        filename = FILENAME_EXPR.format(s=snapshots_dir, idx=idx)

    if to_merge:
        post.merge_analysis(snapshots_dir)

def load(name, params, dyn_vars, plot_stride, start=0):
    populate_globals(params)
    snapshots_dir = SNAPSHOTS_DIR % name
    merge(name)

    solver, domain = get_solver(params)
    z = domain.grid(1, scales=INTERP_Z)

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
            print(solver.sim_time, sim_times[idx])

            for varname in dyn_vars:
                values = solver.state[varname]
                values.set_scales((INTERP_X, INTERP_Z),
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

    if not NL:
        state_vars['ux_z'] = np.gradient(state_vars['ux'], axis=2)

    state_vars['S_{px}'] = RHO0 * np.exp(-z/ H) * (
        (state_vars['ux'] * state_vars['uz']) +
        (NU * np.exp(state_vars['U']) * state_vars['ux_z']))
    return np.array(total_sim_times[start::plot_stride]), domain, state_vars

def plot(name, params):
    rank = CW.rank
    size = CW.size
    populate_globals(params)

    slice_suffix = '(x=0)'
    mean_suffix = '(mean)'
    sub_suffix = ' (- mean)'
    res_suffix = ' (res)'
    snapshots_dir = SNAPSHOTS_DIR % name
    matplotlib.rcParams.update({'font.size': 6})
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
        {
            'save_fmt_str': 's_%03i.png',
            'slice_vars': ['uz', 'ux'],
            'mean_vars': ['S_{px}'],
        },
        {
            'save_fmt_str': 'm_%03i.png',
            'plot_vars': ['uz', 'ux', 'W', 'U'],
        },
    ]

    dyn_vars = ['uz', 'ux', 'U', 'W']
    if NL:
        dyn_vars += ['ux_z']
    sim_times, domain, state_vars = load(name, params, dyn_vars, plot_stride,
        start=0)

    x = domain.grid(0, scales=INTERP_X)
    z = domain.grid(1, scales=INTERP_Z)
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

        uz_est = F * get_uz_f_ratio(params)
        ux_est = uz_est * KZ / KX

        for t_idx, sim_time in list(enumerate(sim_times))[rank::size]:
            fig = plt.figure(dpi=400)

            uz_anal = get_anal_uz(params, sim_time, x, z)
            ux_anal = get_anal_ux(params, sim_time, x, z)
            uz_mean = np.outer(np.ones(N_X),
                               state_vars['uz%s' % mean_suffix][0][t_idx])
            ux_mean = np.outer(np.ones(N_X),
                               state_vars['ux%s' % mean_suffix][0][t_idx])
            S_px_mean = state_vars['S_{px}%s' % mean_suffix][0]
            z_top = get_front_idx(S_px_mean[t_idx],
                                  get_flux_th(params) * 0.2)
            z_bot = get_idx(Z0 + 3 * S, z[0])

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
                              linewidth=0.7,
                              label='Data')
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
                    k_damp = get_k_damp(params)
                    p = axes.plot(
                        uz_est**2 / 2
                            * abs(KZ / KX)
                            * RHO0 * np.exp(-Z0 / H)
                            * np.exp(-k_damp * 2 * (z[0] - Z0)),
                        z[0],
                        'orange',
                        label=r'$x_0z_0$ (Anal.)',
                        linewidth=0.5)
                    rho0 = RHO0 * np.exp(-z / H)
                    # compute all of the S_px cross terms (00 is model)
                    Spx01 = np.sum(rho0 * state_vars['ux'][t_idx] *
                                   (state_vars['uz'][t_idx] - uz_anal),
                                   axis=0) / N_X
                    Spx10 = np.sum(rho0 * state_vars['uz'][t_idx] *
                                   (state_vars['ux'][t_idx] - ux_anal),
                                   axis=0) / N_X
                    Spx11 = np.sum(rho0 * (state_vars['ux'][t_idx] - ux_anal) *
                                   (state_vars['uz'][t_idx] - uz_anal),
                                   axis=0) / N_X
                    p = axes.plot(Spx01[z_bot: z_top],
                                  z[0, z_bot: z_top],
                                  'g:',
                                  linewidth=0.4,
                                  label=r'$x_0z_1$')
                    p = axes.plot(Spx10[z_bot: z_top],
                                  z[0, z_bot: z_top],
                                  'b:',
                                  linewidth=0.4,
                                  label=r'$x_1z_0$')
                    p = axes.plot(Spx11[z_bot: z_top],
                                  z[0, z_bot: z_top],
                                  'k-',
                                  linewidth=0.7,
                                  label=r'$x_1z_1$')
                    axes.legend()

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
                # if var in mean_vars:
                #     p = axes.plot(
                #         var_min[t_idx],
                #         z[0],
                #         'r:',
                #         linewidth=0.2)
                #     p = axes.plot(
                #         var_max[t_idx],
                #         z[0],
                #         'r:',
                #         linewidth=0.2)

                plt.xticks(rotation=30)
                plt.yticks(rotation=30)
                xlims = [var_dat.min(), var_dat.max()]
                axes.set_xlim(*xlims)
                axes.set_ylim(z[0].min(), z[0].max())
                p = axes.plot(xlims,
                              [SPONGE_LOW + SPONGE_WIDTH] * len(xlims),
                              'r:',
                              linewidth=0.5)
                p = axes.plot(xlims,
                              [SPONGE_HIGH - SPONGE_WIDTH] * len(xlims),
                              'r:',
                              linewidth=0.5)
                p = axes.plot(xlims,
                              [Z0 + 3 * S] * len(xlims),
                              'b--',
                              linewidth=0.5)
                p = axes.plot(xlims,
                              [Z0 - 3 * S] * len(xlims),
                              'b--',
                              linewidth=0.5)
                idx += 1
            for var in c_vars:
                axes = fig.add_subplot(n_rows,
                                       n_cols,
                                       idx,
                                       title=r'$%s$ (kx=kx_d)' % var)
                var_dat = state_vars[var]
                kx_idx = round(KX / (2 * np.pi / XMAX))
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
                (name.replace('_', '.'), sim_time, KX, KZ, OMEGA))
            fig.subplots_adjust(hspace=0.7, wspace=0.6)
            savefig = save_fmt_str % (t_idx)
            plt.savefig('%s/%s' % (snapshots_dir, savefig))
            logger.info('Saved %s/%s' % (snapshots_dir, savefig))
            plt.close()

def write_front(name, params):
    ''' few plots for front, defined where flux drops below 1/2 of theory '''
    populate_globals(params)
    u_c = OMEGA / KX
    dyn_vars = ['uz', 'ux', 'U', 'W']
    if NL:
        dyn_vars += ['ux_z']
    snapshots_dir = SNAPSHOTS_DIR % name
    logfile = '%s/data.pkl' % snapshots_dir

    flux_th = get_flux_th(params)
    flux_threshold = flux_th * 0.3

    # generate if does not exist
    if not os.path.exists(logfile):
        print('log file not found, generating')
        sim_times, domain, state_vars = load(
            name, params, dyn_vars, plot_stride=1, start=0)
        x = domain.grid(0, scales=INTERP_X)
        z = domain.grid(1, scales=INTERP_Z)
        xmesh, zmesh = quad_mesh(x=x[:, 0], y=z[0])
        z0 = z[0]
        rho0 = RHO0 * np.exp(-z0 / H)

        ux_res_slice = []
        uz_res_slice = []
        Spx11 = []
        amps = []

        S_px = horiz_mean(state_vars['S_{px}'], N_X)
        u0 = horiz_mean(state_vars['ux'], N_X)
        u0_z = horiz_mean(state_vars['ux_z'], N_X)

        uz_est = F * get_uz_f_ratio(params)
        ux_est = uz_est * KZ / KX
        for t_idx, sim_time in enumerate(sim_times):
            # figure out what to subtract by convolution over lambda_z
            z_bot = get_idx(Z0 + 3 * S, z0)
            z_top = get_idx(Z0 + 3 * S + 2 * np.pi / abs(KZ), z0)

            anal_ux = get_anal_ux(params, sim_time, x, z)[:, z_bot: z_top]
            norm_x = np.outer(np.ones(N_X), np.sum(anal_ux**2, axis=0))
            anal_uz = get_anal_uz(params, sim_time, x, z)[:, z_bot: z_top]
            norm_z = np.outer(np.ones(N_X), np.sum(anal_uz**2, axis=0))
            x_amp = np.sum(state_vars['ux'][t_idx, :, z_bot: z_top]
                           * anal_ux / norm_x) / np.sum(anal_ux**2/ norm_x)
            z_amp = np.sum(state_vars['uz'][t_idx, :, z_bot: z_top]
                           * anal_uz / norm_z) / np.sum(anal_uz**2 / norm_z)
            amp = (x_amp + z_amp) / 2
            print(amp, x_amp, z_amp)
            x_lin_est = amp * (KX / KZ) * get_anal_ux(params, sim_time, x, z)
            z_lin_est = amp * get_anal_uz(params, sim_time, x, z)

            dux = state_vars['ux'][t_idx] - x_lin_est
            duz = state_vars['uz'][t_idx] - z_lin_est

            ux_res = dux / (ux_est * np.exp((z - Z0) / (2 * H)))
            uz_res = dux / (uz_est * np.exp((z - Z0) / (2 * H)))
            slice_idx = get_idx(z0[z_top] - 1 / KZ, z0)

            ux_res_slice.append(ux_res[:, slice_idx - 1])
            uz_res_slice.append(uz_res[:, slice_idx - 1])

            amps.append(amp)
            Spx11.append(np.sum(rho0 * dux * duz, axis=0) / N_X)

        with open(logfile, 'wb') as data:
            pickle.dump((z0, sim_times, S_px, Spx11, u0, u0_z,
                         np.array(amps), ux_res_slice, uz_res_slice), data)
    else:
        print('log file found, not regenerating')

def plot_front(name, params):
    populate_globals(params)
    N = np.sqrt(g / H)
    u_c = OMEGA / KX
    V_GZ = abs(get_vgz(g, H, KX, KZ))
    flux_th = get_flux_th(params)
    flux_threshold = flux_th * 0.3
    k_damp = get_k_damp(params)

    dyn_vars = ['uz', 'ux', 'U', 'W']
    snapshots_dir = SNAPSHOTS_DIR % name
    logfile = '%s/data.pkl' % snapshots_dir
    if NL:
        dyn_vars += ['ux_z']
    if not os.path.exists(logfile):
        write_front(name, params)

    print('Loading data')
    with open(logfile, 'rb') as data:
        z0, sim_times, S_px, Spx11, u0, u0_z, \
            amps, ux_res_slice, uz_res_slice = pickle.load(data)
        Spx11 = np.array(Spx11) / flux_th

    tf = sim_times[-1]
    start_idx = get_idx(200, sim_times)
    t = sim_times[start_idx: ]

    S_px0 = [] # S_px averaged near origin
    dSpx = [] # Delta S_px using S criterion
    front_pos = [] # front position using S criterion
    front_idxs = []
    dz = abs(1 / KZ)
    l_z = abs(2 * np.pi / KZ)
    z_b = Z0 + 3 * S
    z_b_idx = get_idx(z_b, z0)
    for t_idx, sim_time in enumerate(sim_times):
        front_idx = get_front_idx(S_px[t_idx], flux_threshold)
        z_c = z0[front_idx]
        front_pos.append(z_c)
        front_idxs.append(front_idx)

        # measure flux incident at dz critical layer
        dz_idx = get_idx(dz, z0)
        S_px0.append(-np.mean(S_px[
            t_idx, get_idx(z_b, z0): get_idx(z_b + l_z, z0)]))
        dSpx.append(-np.mean(S_px[
            t_idx, get_idx(z_c - l_z - dz, z0): get_idx(z_c - dz, z0)]))

    times = get_times([1/8, 3/8, 5/8, 7/8], sim_times, start_idx)
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
                 np.exp(-k_damp * 2 * (z0_cut - Z0)),
                 linewidth=1.5,
                 label=r'Model')
        plt.xlim(z_b, ZMAX)
        plt.ylim(-0.1, 1.1)
        plt.legend(fontsize=6)

        plt.xlabel(r'$z(H)$')
        plt.ylabel(r'$S_{px} / S_0$')
        plt.savefig('%s/fluxes.png' % snapshots_dir, dpi=400)
        plt.close()

        #####################################################################
        # f_amps.png
        #
        # convolved amplitudes over time
        #####################################################################
        plt.plot(t,
                 amps[start_idx: ],
                 label=r'$A$')
        plt.legend(fontsize=6)
        plt.xlabel(r'$t (N^{-1})$')
        plt.savefig('%s/f_amps.png' % snapshots_dir, dpi=400)
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
                     linewidth=0.7)
        # text showing min Ri
        ax1.text(z_b + 0.2, 0.8,
                 'Min Ri: %.3f' % (N / u0_z.max())**2,
                 fontsize=8)
        # overlay analytical flux including viscous dissipation
        ax2.plot(z0_cut,
                 np.exp(-k_damp * 2 * (z0_cut - Z0)),
                 linewidth=1.5,
                 label=r'$\nu$-only')
        # indicate vertical averaging wavelength
        ax2.axvspan(front_pos[times[-1]] - dz - l_z,
                    front_pos[times[-1]] - dz,
                    color='grey')
        ax1.set_xlim(z_b, ZMAX)
        ax2.set_xlim(z_b, ZMAX)
        ax1.set_ylim(-0.1, 1.25)
        ax2.set_ylim(-0.1, 1.1)
        ax2.legend(fontsize=6)

        ax1.set_ylabel(r'$U_0 / c_{ph, x}$')
        ax2.set_ylabel(r'$S_{px} / S_0$')
        ax2.set_xlabel(r'$z(H)$')
        plt.savefig('%s/fluxes.png' % snapshots_dir, dpi=400)
        plt.close()

        #####################################################################
        # fluxes_time.png
        #
        # plot breakdown of fluxes at a few times
        #####################################################################
        times = get_times([1/8, 1/5, 7/8], sim_times, start_idx)
        f, ax_lst = plt.subplots(len(times), 1, sharex=True)
        f.subplots_adjust(hspace=0)

        # plot residuals at a few times
        for ax, time in zip(ax_lst, times):
            z_t_idx = get_front_idx(S_px[time],
                                    get_flux_th(params) * 0.2)
            ax.plot(z0_cut,
                    np.exp(-k_damp * 2 * (z0_cut - Z0)),
                    linewidth=1.5,
                    label=r'$u_{x0}u_{z0}$')
            Spx_avg = np.sum(S_px[time - 2: time + 2, z_b_idx: ], axis=0) / 4
            ax.plot(z0_cut,
                    Spx_avg / flux_th,
                    linewidth=0.7,
                    label=r'$u_xu_z$')
            Spx_avg = np.sum(Spx11[time - 2: time + 2, : ], axis=0) / 4
            ax.plot(z0[z_b_idx: z_t_idx],
                    Spx_avg[z_b_idx: z_t_idx],
                    linewidth=0.7,
                    label=r'$\delta u_x \delta u_z$')
            ax.legend(fontsize=6, loc='upper right')
            ax.set_xlim(z_b, ZMAX)
            ax.set_ylim(-0.6, 1.2)
            ax.set_ylabel(r'$S_{px, t=%.1f} / S_0$' % sim_times[time])
        ax_lst[-1].set_xlabel(r'$z(H)$')
        Spx11_mean = []
        for time in range(start_idx, len(sim_times)):
            z_t_idx = get_front_idx(S_px[time], get_flux_th(params) * 0.3)
            half_idx = (z_t_idx + z_b_idx) // 2
            try:
                Spx11_last = Spx11[time, z_b_idx: half_idx]
                Spx11_mean.append(np.mean(Spx11_last))
            except: # sometimes front idx is weird spot?
                continue
        plt.savefig('%s/fluxes_time.png' % snapshots_dir, dpi=400)
        plt.close()

        #####################################################################
        # front.png
        #
        # plot front position and absorbed flux over time
        #####################################################################
        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        f.subplots_adjust(hspace=0)

        zf = np.max(front_pos[-5: -1])

        S_px0 = np.array(S_px0)
        dSpx = np.array(dSpx)
        front_pos = np.array(front_pos)

        # compute front position
        front_vel_S = dSpx / (RHO0 * np.exp(-front_pos / H) * u_c)
        front_pos_intg_S = np.cumsum((front_vel_S * np.gradient(sim_times))
                                      [start_idx: ])
        front_pos_intg_S += zf - front_pos_intg_S[-1]

        # estimate incident Delta S_px if all from S_px0 that's viscously
        # damped, compare to other Delta S_px criteria/from data
        color_idx = 0
        dSpx0 = -S_px0[start_idx: ] / flux_th * \
            np.exp(-k_damp * 2 * (front_pos[start_idx: ] - (z_b + l_z / 2)))
        ax1.plot(t,
                 -S_px0[start_idx: ] / flux_th,
                 '%s-' % PLT_COLORS[color_idx],
                 label=r'$\Delta S_{px,0}|_{z=z_0}$',
                 linewidth=0.7)
        color_idx += 1
        ax1.plot(t,
                 dSpx0,
                 '%s-' % PLT_COLORS[color_idx],
                 label=r'$\Delta S_{px,0}|_{z=z_{c}}$',
                 linewidth=0.7)
        color_idx += 1
        ax1.plot(t,
                 -dSpx[start_idx: ] / flux_th,
                 '%s-' % PLT_COLORS[color_idx],
                 label=r'$\Delta S_{px}(z_{c})$',
                 linewidth=0.7)
        color_idx += 1
        ax1.set_ylabel(r'$S_{px} / S_{px, 0}$')
        ax1.legend(fontsize=6)

        # compare forecasts of front position using two predictors integrated
        # from incident flux in data
        color_idx = 0
        ax2.plot(t,
                 front_pos_intg_S,
                 '%s-' % PLT_COLORS[color_idx],
                 label='Model (data $\Delta S_{px}(z_{c})$)',
                 linewidth=0.7)
        color_idx += 1
        ax2.plot(t,
                 front_pos[start_idx: ],
                 '%s-' % PLT_COLORS[color_idx],
                 label='Data (S)',
                 linewidth=0.7)
        color_idx += 1

        # three multipliers are (i) average incident flux, (ii) estimated
        # incident flux extrapolated from nu and (iii) full flux
        mean_incident = -np.mean(dSpx[len(dSpx) // 8: ])
        est_generated_flux = -np.mean(S_px0[len(S_px0) // 8: ])
        mean_pos = np.max(front_pos[len(front_pos) // 3: ])
        est_incident_flux = est_generated_flux *\
            np.exp(-k_damp * 2 * (mean_pos - Z0))
        flux_mults = [mean_incident / flux_th,
                      est_incident_flux / flux_th,
                      1]
        for mult in flux_mults:
            tau = H * RHO0 * u_c / (flux_th * mult)
            pos_anal = -H * np.log(
                (t - tf + tau * np.exp(-zf/H))
                / tau)
            ax2.plot(t,
                     pos_anal,
                     '%s:' % PLT_COLORS[color_idx],
                     label='Model ($%.2f S_{px,0}$)' % mult,
                     linewidth=0.7)
            color_idx += 1
        ax2.set_ylabel(r'$z_c$')
        ax2.set_xlabel(r't')
        ax2.set_ylim([zf, 10])
        ax2.legend(fontsize=6)
        plt.savefig('%s/front.png' % snapshots_dir, dpi=400)
        plt.close()

        #####################################################################
        # f_refl.png
        #
        # reflection coeff calculations
        #####################################################################
        f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
        f.subplots_adjust(hspace=0)

        # dSpx0 is visc-extrapolated flux, time shift it and compare to amps
        # extrapolate 3/4 of distance, convolution is evenly weighted between
        # z_b, z_c
        prop_time = (front_pos[start_idx: ] - Z0 - np.pi / KZ) / V_GZ
        S_excited = (amps**2)[start_idx: ] * \
            np.exp(-k_damp * 2 * (front_pos[start_idx: ] - (z_b + l_z / 2)))
        ax1.plot(t,
                 S_excited,
                 'g:',
                 label='Incident',
                 linewidth=0.7)
        ax1.plot(t + prop_time,
                 S_excited,
                 'b:',
                 label=r'Incident + $\frac{\Delta z}{c_{g,z}}$',
                 linewidth=0.7)
        ax1.plot(t,
                 -dSpx[start_idx: ] / flux_th,
                 'k:',
                 label='Absorbed',
                 linewidth=1.0)
        ax1.set_ylabel(r'$S_{px} / S_0$')
        ax1.legend(fontsize=6)
        shifted_dS = interp1d(t + prop_time, S_excited)
        absorbed_dS = interp1d(t, -dSpx[start_idx: ] / flux_th)
        t_refl = np.linspace((t + prop_time)[0], t[-1], len(t))
        refl = [(shifted_dS(t) - absorbed_dS(t)) / shifted_dS(t)
                for t in t_refl]

        ax2.plot(t_refl, refl, 'r:', linewidth=0.7)

        ax2.text(t[0],
                 0.85 * ax2.get_ylim()[0] + 0.15 * ax2.get_ylim()[1],
                 'Mean: (%.3f)' % np.mean(refl),
                 fontsize=8)
        ax2.set_ylabel(r'Reflectivity')
        ax2.set_xlabel(r'$t$')

        ri_vals = np.array([(N / u0_z[idx + start_idx, z_idx])**2
                            for idx, z_idx in
                                enumerate(front_idxs[start_idx: ])])
        ax3.plot(t, ri_vals)
        ax3.set_ylim([0, 3])
        ax3.set_ylabel(r"Ri $(N / U0')^2$")
        ax3.text(t[0], 0.5,
                 'Min Ri: %.3f' % ri_vals.min(),
                 fontsize=8)

        plt.savefig('%s/f_refl.png' % snapshots_dir, dpi=400)
        plt.close()

        #####################################################################
        # f_amps.png
        #
        # convolved amplitudes over time
        #####################################################################
        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        f.subplots_adjust(hspace=0)
        ax1.plot(t,
                 amps[start_idx: ],
                 label=r'$A$')
        ax1.legend(fontsize=6)
        u0_dat = u0[start_idx:, get_idx(Z0, z0)] / (OMEGA / KX)
        ax2.plot(t, u0_dat)
        ax2.set_ylabel(r'$\bar{U}_0 / c_{ph,x}$')
        ax2.set_xlabel(r'$t (N^{-1})$')
        plt.savefig('%s/f_amps.png' % snapshots_dir, dpi=400)
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
    kx = np.linspace(0, 2 * np.pi * (N_X // 2) / XMAX,
                     N_X // 2)[1: ]
    # drop endpoints
    for idx in np.linspace(0, len(ux_ffts), num_plots + 2)[1: -1]:
        idx = int(idx)
        x_avg = np.mean(ux_ffts[idx - half_avg:idx + half_avg, :], 0)
        z_avg = np.mean(uz_ffts[idx - half_avg:idx + half_avg, :], 0)
        ax1.loglog(kx, x_avg, label='t=%.1f' % sim_times[idx], linewidth=0.7)
        ax2.loglog(kx, z_avg, label='t=%.1f' % sim_times[idx], linewidth=0.7)
    if NU > 0:
        visc_kx = np.sqrt(OMEGA / (2 * NU))
        ax1.axvline(x=visc_kx, linewidth=1.5, color='red')
        ax2.axvline(x=visc_kx, linewidth=1.5, color='red')
    ax1.set_ylabel(r'$\tilde{u}_x(k_x)$')
    ax2.set_ylabel(r'$\tilde{u}_z(k_x)$')
    ax2.set_xlabel(r'$k_x$')
    ax1.legend(fontsize=6)
    ax2.legend(fontsize=6)
    plt.savefig('%s/fft.png' % snapshots_dir, dpi=400)
    plt.close()

