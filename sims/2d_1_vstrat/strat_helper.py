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
plot_stride = 15

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
                2 * RHO0 * OMEGA**2 * KZ)

def get_flux_th(params):
    return (F * get_uz_f_ratio(params))**2 / 2 * abs(KZ / KX) * RHO0

def get_k_damp(params):
    populate_globals(params)
    k = np.sqrt(KX**2 + KZ**2)

    return NU * k**5 / abs(KZ * g / H * KX)

def get_anal_uz(params, t, x, z):
    populate_globals(params)
    uz_est = F * get_uz_f_ratio(params)
    k_damp = get_k_damp(params)

    return uz_est * (
        np.exp(-k_damp
                 * (z - Z0))
        * np.sin(KX * x + KZ * (z - Z0) - OMEGA * t
                 + 1 / (2 * KZ * H)))

def get_anal_ux(params, t, x, z):
    populate_globals(params)
    ux_est = F * get_uz_f_ratio(params) * KZ / KX
    k_damp = get_k_damp(params)

    return ux_est * (
        np.exp(-k_damp * (z[0] - Z0))
        * np.sin(KX * x + KZ * (z - Z0) - OMEGA * t
                 + 1 / (KZ * H)))

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

    ux = solver.state['ux']
    z = domain.grid(1)

    # turns on at Z0 + ZMAX / 2 w/ width lambda_z / 2 * pi, turns off at sponge zone
    z_bot = (Z0 + ZMAX) * 0.4
    width = abs(1.5 / KZ)
    z_top = SPONGE_HIGH - 3 * (ZMAX - SPONGE_HIGH) * SPONGE_WIDTH
    ux['g'] = OMEGA / KX * UZ0_COEFF * (
        np.tanh((z - z_bot) / width) -
        np.tanh((z - z_top) / (SPONGE_WIDTH * (ZMAX - SPONGE_HIGH)))) / 2
    print(z_bot, z_top, width, ZMAX )
    plt.plot(z[0, :], ux['g'][0, :])
    plt.savefig('abc.png')
    raise ValueError('foo');
    return 0, DT


def get_solver(params):
    ''' sets up solver '''
    x_basis = de.Fourier('x',
                         N_X // INTERP_X,
                         interval=(0, XMAX),
                         dealias=3/2)
    z_basis = de.Fourier('z',
                         N_Z // INTERP_Z,
                         interval=(0, ZMAX),
                         dealias=3/2)
    domain = de.Domain([x_basis, z_basis], np.float64)

    problem = de.IVP(domain, variables=['P', 'rho', 'ux', 'uz'])
    problem.parameters.update(params)

    problem.substitutions['sponge'] = 'SPONGE_STRENGTH * 0.5 * ' +\
        '(2 + tanh((z - SPONGE_HIGH) / (SPONGE_WIDTH * (ZMAX - SPONGE_HIGH))) - ' +\
        'tanh((z - SPONGE_LOW) / (SPONGE_WIDTH * (SPONGE_LOW))))'
    problem.substitutions['mask'] = \
        '0.5 * (1 + tanh((z - (Z0 + 3 * S)) / (S / 2)))'
    # problem.substitutions['mask'] = '1'
    problem.add_equation('dx(ux) + dz(uz) = 0', condition='nx != 0 or nz != 0')
    problem.add_equation(
        'dt(rho) - RHO0 * uz / H' +
        '- NU * (N_Z/N_X)**6 * dx(dx(dx(dx(dx(dx(rho))))))' +
        '- NU * dz(dz(dz(dz(dz(dz(rho))))))' +
        '= -sponge * rho' +
        '- mask * (ux * dx(rho) + uz * dz(rho))' +
        '+ F * exp(-(z - Z0)**2 / (2 * S**2)) *cos(KX * x - OMEGA * t)')
    problem.add_equation(
        'dt(ux) + dx(P) / RHO0' +
        '- NU * (N_Z/N_X)**6 * dx(dx(dx(dx(dx(dx(ux))))))' +
        '- NU * dz(dz(dz(dz(dz(dz(ux))))))' +
        '= - sponge * ux' +
        '- mask * (ux * dx(ux) + uz * dz(ux))')
    problem.add_equation(
        'dt(uz) + dz(P) / RHO0 + rho * g / RHO0' +
        '- NU * (N_Z/N_X)**6 * dx(dx(dx(dx(dx(dx(uz))))))' +
        '- NU * dz(dz(dz(dz(dz(dz(uz))))))' +
        '= -sponge * uz' +
        '- mask * (ux * dx(uz) + uz * dz(uz))')
    problem.add_equation('P = 0', condition='nx == 0 and nz == 0')

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
    snapshots.add_task("integ(RHO0 * ux * uz, 'x') / XMAX", name='S_{px}')

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
            logger.info('Max u = %f' %flow.max('u'))

def merge(name):
    snapshots_dir = SNAPSHOTS_DIR % name
    filename = FILENAME_EXPR.format(s=snapshots_dir, idx=1)
    if not os.path.exists(filename):
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

    state_vars['rho'] += RHO0
    state_vars['rho1'] = state_vars['rho'] - RHO0
    state_vars['P1'] = state_vars['P']
    # compatibility
    state_vars['ux_z'] = np.gradient(state_vars['ux'], axis=-1) * \
        N_Z / ZMAX

    state_vars['S_{px}'] = state_vars['rho'] * (state_vars['ux'] *
                                              state_vars['uz'])
    return sim_times[start::plot_stride], domain, state_vars

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
    # z_b = N_Z // 4
    z_b = 0

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
            'slice_vars': ['uz'],
            'mean_vars': ['ux', 'S_{px}'],
            'res_vars_vars': ['uz'],
        },
        {
            'save_fmt_str': 'm_%03i.png',
            'plot_vars': ['ux', 'uz', 'rho1', 'P1'],
        },
    ]

    dyn_vars = ['uz', 'ux', 'rho', 'P']
    sim_times, domain, state_vars = load(name, params, dyn_vars, plot_stride)

    x = domain.grid(0, scales=INTERP_X)
    z = domain.grid(1, scales=INTERP_Z)[: , z_b:]
    xmesh, zmesh = quad_mesh(x=x[:, 0], y=z[0])
    x2mesh, z2mesh = quad_mesh(x=np.arange(N_X // 2), y=z[0])

    # preprocess
    for var in dyn_vars + ['S_{px}', 'ux_z']:
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
                            / uz_est
                        title = 'u_z'
                    elif var == 'ux%s' % res_suffix:
                        var_dat = (state_vars['ux'][t_idx] - ux_anal - ux_mean)\
                            / ux_est
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

                    var_dat = state_vars[var][t_idx, :, z_b: ]
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

                var_dat = state_vars[var][:, : , z_b:]
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
                    var_dat = state_vars[var][:, z_b:]
                else:
                    var_dat = state_vars[var][0][:, z_b:]
                    var_min = state_vars[var][1][:, z_b:]
                    var_max = state_vars[var][2][:, z_b:]

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
                        -uz_est * (0 * z[0] + 1), z[0], 'g',
                        uz_est * (0 * z[0] + 1), z[0], 'g',
                        linewidth=0.5)

                if var == 'ux%s' % slice_suffix:
                    p = axes.plot(
                        ux_anal[0, :],
                        z[0],
                        'orange',
                        linewidth=0.5)
                    p = axes.plot(
                        -ux_est, z[0], 'g',
                        ux_est, z[0], 'g',
                        linewidth=0.5)

                if var == 'S_{px}%s' % mean_suffix:
                    k_damp = get_k_damp(params)
                    p = axes.plot(
                        uz_est**2 / 2
                            * abs(KZ / KX)
                            * RHO0
                            * np.exp(-k_damp * 2 * (z[0] - Z0)),
                        z[0],
                        'orange',
                        label=r'$x_0z_0$ (Anal.)',
                        linewidth=0.5)
                    # compute all of the S_px cross terms (00 is model)
                    Spx01 = np.sum(RHO0 * state_vars['ux'][t_idx] *
                                   (state_vars['uz'][t_idx] - uz_anal),
                                   axis=0) / N_X
                    Spx10 = np.sum(RHO0 * state_vars['uz'][t_idx] *
                                   (state_vars['ux'][t_idx] - ux_anal),
                                   axis=0) / N_X
                    Spx11 = np.sum(RHO0 * (state_vars['ux'][t_idx] - ux_anal) *
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
                              [SPONGE_LOW + SPONGE_WIDTH * SPONGE_LOW]\
                                * len(xlims),
                              'r:',
                              linewidth=0.5)
                p = axes.plot(xlims,
                              [SPONGE_HIGH - SPONGE_WIDTH *
                               (ZMAX - SPONGE_HIGH)] * len(xlims),
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
    dyn_vars = ['uz', 'ux', 'rho', 'P']
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

        ux_res_slice = []
        uz_res_slice = []
        Spx01 = []
        Spx10 = []
        Spx11 = []
        x_amps = []
        z_amps = []

        S_px = horiz_mean(state_vars['S_{px}'], N_X)
        u0 = horiz_mean(state_vars['ux'], N_X)

        uz_est = F * get_uz_f_ratio(params)
        ux_est = uz_est * KZ / KX
        for t_idx, sim_time in enumerate(sim_times):
            # figure out what to subtract by convolution
            front_idx = get_front_idx(S_px[t_idx], flux_threshold)
            z_bot = get_idx(Z0 + 3 * S, z0)

            front_idx = (front_idx + z_bot) // 2 # convolve only over bottom half
            anal_ux = get_anal_ux(params, sim_time, x, z)[:, z_bot: front_idx]
            anal_uz = get_anal_uz(params, sim_time, x, z)[:, z_bot: front_idx]
            x_amp = np.sum(state_vars['ux'][t_idx, :, z_bot: front_idx]
                           * anal_ux) / np.sum(anal_ux**2)
            z_amp = np.sum(state_vars['uz'][t_idx, :, z_bot: front_idx]
                           * anal_uz) / np.sum(anal_uz**2)
            x_lin_est = x_amp * get_anal_ux(params, sim_time, x, z)
            z_lin_est = z_amp * get_anal_uz(params, sim_time, x, z)

            dux = state_vars['ux'][t_idx] - x_lin_est
            duz = state_vars['uz'][t_idx] - z_lin_est

            ux_res = dux / (ux_est * np.exp((z - Z0) / (2 * H)))
            uz_res = dux / (uz_est * np.exp((z - Z0) / (2 * H)))
            slice_idx = get_idx(z0[front_idx] - 1 / KZ, z0)

            ux_res_slice.append(ux_res[:, slice_idx - 1])
            uz_res_slice.append(uz_res[:, slice_idx - 1])

            x_amps.append(x_amp)
            z_amps.append(z_amp)
            Spx01.append(np.sum(RHO0 * x_lin_est * duz, axis=0) / N_X)
            Spx10.append(np.sum(RHO0 * z_lin_est * dux, axis=0) / N_X)
            Spx11.append(np.sum(RHO0 * dux * duz, axis=0) / N_X)

        with open(logfile, 'wb') as data:
            pickle.dump((z0, sim_times, S_px, Spx01, Spx10, Spx11,
                         x_amps, z_amps, u0, ux_res_slice, uz_res_slice), data)
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

    dyn_vars = ['uz', 'ux', 'rho', 'P']
    snapshots_dir = SNAPSHOTS_DIR % name
    logfile = '%s/data.pkl' % snapshots_dir
    if NL:
        dyn_vars += ['ux_z']
    if not os.path.exists(logfile):
        write_front(name, params)

    print('Loading data')
    with open(logfile, 'rb') as data:
        z0, sim_times, S_px, Spx01, Spx10, Spx11, x_amps, z_amps, \
            u0, ux_res_slice, uz_res_slice = pickle.load(data)
        Spx01 = np.array(Spx01) / flux_th
        Spx10 = np.array(Spx10) / flux_th
        Spx11 = np.array(Spx11) / flux_th
        x_amps = np.array(x_amps)
        z_amps = np.array(z_amps)

    tf = sim_times[-1]
    start_idx = get_idx(600, sim_times)
    t = sim_times[start_idx: ]
    dt = np.mean(np.gradient(sim_times[1: -1])) # should be T_F / NUM_SNAPSHOTS

    S_px0 = [] # S_px averaged near origin
    dSpx_S = [] # Delta S_px using S criterion
    front_pos_S = [] # front position using S criterion
    dz = abs(1 / KZ)
    l_z = abs(2 * np.pi / KZ)
    z_b = Z0 + 3 * S
    z_b_idx = get_idx(z_b, z0)
    for t_idx, sim_time in enumerate(sim_times):
        front_idx_S = get_front_idx(S_px[t_idx], flux_threshold)
        z_cS = z0[front_idx_S]
        front_pos_S.append(z_cS)

        # measure flux incident at dz critical layer
        dz_idx = get_idx(dz, z0)
        S_px0.append(-np.mean(S_px[
            t_idx, get_idx(z_b, z0): get_idx(z_b + l_z, z0)]))
        dSpx_S.append(-np.mean(S_px[
            t_idx, get_idx(z_cS - l_z - dz, z0): get_idx(z_cS - dz, z0)]))

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
        u0_z = np.gradient(u0[:, z_b_idx: ], axis=1) /\
            np.outer(np.ones(len(sim_times)), np.gradient(z0[z_b_idx: ]))
        ax1.text(z_b + 0.2, 0.8,
                 'Min Ri: %.3f' % (N / u0_z.max())**2)
        # overlay analytical flux including viscous dissipation
        ax2.plot(z0_cut,
                 np.exp(-k_damp * 2 * (z0_cut - Z0)),
                 linewidth=1.5,
                 label=r'$\nu$-only')
        # indicate vertical averaging wavelength
        ax2.axvspan(front_pos_S[times[-1]] - dz - l_z,
                    front_pos_S[times[-1]] - dz,
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
            for lbl, Spx in zip([r'$u_x\delta u_z$', r'$\delta u_x u_z$',
                                 r'$\delta u_x \delta u_z$'],
                                [Spx01, Spx10, Spx11]):
                Spx_avg = np.sum(Spx[time - 2: time + 2, : ], axis=0) / 4
                ax.plot(z0[z_b_idx: z_t_idx],
                        Spx_avg[z_b_idx: z_t_idx],
                        linewidth=0.7,
                        label=lbl)
            ax.legend(fontsize=6, loc='upper right')
            ax.set_xlim(z_b, ZMAX)
            ax.set_ylim(-0.6, 1.2)
            ax.set_ylabel(r'$S_{px, t=%.1f} / S_0$' % sim_times[time])
        ax_lst[-1].set_xlabel(r'$z(H)$')
        Spx01_amp = []
        Spx10_amp = []
        Spx11_mean = []
        for time in range(start_idx, len(sim_times)):
            z_t_idx = get_front_idx(S_px[time], get_flux_th(params) * 0.3)
            half_idx = (z_t_idx + z_b_idx) // 2
            try:
                Spx01_last = Spx01[time, z_b_idx: half_idx]
                Spx01_amp.append((np.max(Spx01_last) - np.min(Spx01_last)) / 2)
                Spx10_last = Spx10[time, z_b_idx: half_idx]
                Spx10_amp.append((np.max(Spx10_last) - np.min(Spx10_last)) / 2)
                Spx11_last = Spx11[time, z_b_idx: half_idx]
                Spx11_mean.append(np.mean(Spx11_last))
            except: # sometimes front idx is weird spot?
                continue
        print('%s: Amp 01 = %.3f, Amp 10 = %.3f, Mean 11 = %.3f' %
              (name,
               np.mean(Spx01_amp),
               np.mean(Spx10_amp),
               np.mean(Spx11_mean)))
        plt.savefig('%s/fluxes_time.png' % snapshots_dir, dpi=400)
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
        front_pos_S = np.array(front_pos_S)

        # compute front position
        front_vel_S = dSpx_S / (RHO0 * np.exp(-front_pos_S / H) * u_c)
        front_pos_intg_S = np.cumsum(front_vel_S[start_idx: ]) * dt
        front_pos_intg_S += zf - front_pos_intg_S[-1]

        # estimate incident Delta S_px if all from S_px0 that's viscously
        # damped, compare to other Delta S_px criteria/from data
        color_idx = 0
        dSpx0 = -S_px0[start_idx: ] / flux_th * \
            np.exp(-k_damp * 2 * (front_pos_S[start_idx: ] - (z_b + l_z / 2)))
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
                 -dSpx_S[start_idx: ] / flux_th,
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
                 front_pos_S[start_idx: ],
                 '%s-' % PLT_COLORS[color_idx],
                 label='Data (S)',
                 linewidth=0.7)
        color_idx += 1

        # three multipliers are (i) average incident flux, (ii) estimated
        # incident flux extrapolated from nu and (iii) full flux
        mean_incident = -np.mean(dSpx_S[len(dSpx_S) // 8: ])
        est_generated_flux = -np.mean(S_px0[len(S_px0) // 8: ])
        mean_pos = np.max(front_pos_S[len(front_pos_S) // 3: ])
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
        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        f.subplots_adjust(hspace=0)

        # dSpx0 is visc-extrapolated flux, time shift it and compare to amps
        prop_time = (front_pos_S[start_idx: ] - Z0) / V_GZ
        S_excited = (x_amps * z_amps)[start_idx: ] * \
            np.exp(-k_damp * 2 * (front_pos_S[start_idx: ] - (z_b + l_z / 2)))
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
                 -dSpx_S[start_idx: ] / flux_th,
                 'k:',
                 label='Absorbed',
                 linewidth=1.0)
        ax1.set_ylabel(r'$S_{px} / S_0$')
        ax1.legend(fontsize=6)
        # seems prop_time is twice what it should be...
        # use interp to compare since different t values
        shifted_dS = interp1d(t + prop_time, S_excited)
        shifted_dS2 = interp1d(t + prop_time / 2, S_excited)
        absorbed_dS = interp1d(t, -dSpx_S[start_idx: ] / flux_th)
        t_refl = np.linspace((t + prop_time)[0], t[-1], len(t))
        refl = [(shifted_dS(t) - absorbed_dS(t)) / shifted_dS(t)
                for t in t_refl]
        refl2 = [(shifted_dS2(t) - absorbed_dS(t)) / shifted_dS2(t)
                 for t in t_refl]

        ax2.plot(t_refl, refl, 'r:', label='Full time', linewidth=0.7)
        ax2.plot(t_refl, refl2, 'g', label='Half time', linewidth=1.0)

        ax2.text(t[0], 0.85 * ax2.get_ylim()[0] + 0.15 * ax2.get_ylim()[1],
                 'Means: (%.3f, %.3f)' % (np.mean(refl), np.mean(refl2)))
        ax2.set_ylabel(r'Reflectivity')
        ax2.legend(fontsize=6)

        ax2.set_xlabel(r'$t$')
        plt.savefig('%s/f_refl.png' % snapshots_dir, dpi=400)
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
    # if NU > 0:
    #     visc_kx = np.sqrt(OMEGA / (2 * NU))
    #     ax1.axvline(x=visc_kx, linewidth=1.5, color='red')
    #     ax2.axvline(x=visc_kx, linewidth=1.5, color='red')
    ax1.set_ylabel(r'$\tilde{u}_x(k_x)$')
    ax2.set_ylabel(r'$\tilde{u}_z(k_x)$')
    ax2.set_xlabel(r'$k_x$')
    ax1.legend(fontsize=6)
    ax2.legend(fontsize=6)
    plt.savefig('%s/fft.png' % snapshots_dir, dpi=400)
    plt.close()

    #####################################################################
    # f_amps.png
    #
    # convolved amplitudes over time
    #####################################################################
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    f.subplots_adjust(hspace=0)
    ax1.plot(t,
             x_amps[start_idx: ],
             label=r'$u_x / u_{x0}$')
    ax1.plot(t,
             z_amps[start_idx: ],
             label=r'$u_z / u_{z0}$')
    ax1.legend(fontsize=6)
    ax1.set_ylabel(r'$u / u_0$')
    u0_dat = u0[start_idx:, get_idx(Z0, z0)] / (OMEGA / KX)
    ax2.plot(t, u0_dat)
    ax2.set_ylabel(r'$\bar{U}_0 / c_{ph,x}$')
    ax2.set_xlabel(r'$t$')
    ax2.set_ylim([min(u0_dat) * 1.1, 0])
    plt.savefig('%s/f_amps.png' % snapshots_dir, dpi=400)
    plt.close()
