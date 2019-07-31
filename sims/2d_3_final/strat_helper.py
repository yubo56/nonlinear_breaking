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
from scipy.interpolate import interp1d
from scipy.optimize import minimize

from dedalus import public as de
from dedalus.tools import post
from dedalus.extras.flow_tools import CFL, GlobalFlowProperty
from dedalus.extras.plot_tools import quad_mesh, pad_limits
from mpi4py import MPI
CW = MPI.COMM_WORLD
PLT_COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'];

SNAPSHOTS_DIR = 'snapshots_%s'
FILENAME_EXPR = '{s}/{s}_s{idx}.h5'
Z_TOP_MULT = 1
STRIDE = 15
AVG_IDX = 4
FONTSIZE = 12
DPI=600

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=FONTSIZE)

def populate_globals(var_dict):
    for key, val in var_dict.items():
        exec(key + "=" + repr(val), globals())

def get_omega(g, h, kx, kz):
    return np.sqrt((g / h) * kx**2 / (kx**2 + kz**2 + 0.25 / h**2))

def get_vpz(g, h, kx, kz):
    return -get_omega(g, h, kx, kz) / kz

def get_vgz(g, h, kx, kz):
    return -get_omega(g, h, kx, kz) * kx * kz /\
        (kx**2 + kz**2 + 0.25 / h**2)**(3/2)

def get_idx(z, z0):
    return int(len(np.where(z0 < z)[0]))

def get_front_idx(_S_px, t_idx, z0, flux_threshold):
    '''
    bracket the front location as the average of:
    1) uppermost point where flux exceeds threshold,
    2) lowermost point where flux is still below threshold (above forcing zone)
    '''
    # examine time-averaged
    S_px = np.mean(_S_px[max(0, t_idx - AVG_IDX):
                         min(len(_S_px), t_idx + AVG_IDX)], axis=0)

    max_idx = len(S_px) - 1
    while S_px[max_idx] < flux_threshold and max_idx >= 0:
        max_idx -= 1
    min_idx = get_idx(Z0 + 3 * S, z0)
    while S_px[min_idx] > flux_threshold and min_idx < max_idx:
        min_idx += 1
    return (min_idx + max_idx) // 2

def horiz_mean(field, n_x, axis=1):
    ''' horizontal direction (axis = 1) is uniform spacing, just mean '''
    return np.sum(field, axis=axis) / n_x

def smooth(f):
    # use much stronger smoothing since no longer using time propagation

    # _kernel = np.array([1, 2, 3, 3, 3, 2, 1])
    _kernel = np.array([1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1])
    kernel = _kernel / sum(_kernel)
    f_padded = np.concatenate(([f[0]] * (len(kernel) // 2),
                               f,
                               [f[-1]] * (len(kernel) // 2)))
    return np.convolve(f_padded, kernel, mode='valid')

def get_uz_f_ratio(params):
    ''' get uz(z = z0) / F '''
    return (np.sqrt(2 * np.pi) * params['S'] * params['g'] *
            params['KX']**2) * np.exp(-params['S']**2 * params['KZ']**2/2) / (
                2 * params['RHO0'] * np.exp(-params['Z0'] / params['H'])
                * params['OMEGA']**2 * params['KZ'])

def get_flux_th(params):
    return (params['F'] * get_uz_f_ratio(params))**2 / 2 \
        * abs(params['KZ'] / params['KX']) * params['RHO0']\
        * np.exp(-params['Z0'] / params['H'])

def get_k_damp(params):
    k = np.sqrt(params['KX']**2 + params['KZ']**2)

    return params['NU'] * k**5 /\
        abs(params['KZ'] * params['g'] / params['H'] * params['KX'])

def get_anal_uz(params, t, x, z, phi=0):
    uz_est = params['F'] * get_uz_f_ratio(params)
    k_damp = get_k_damp(params)

    return uz_est * (
        -np.exp((z - params['Z0']) / 2 * params['H'])
        * np.exp(-k_damp * (z - params['Z0']))
        * np.sin(params['KX'] * x
                 + params['KZ'] * (z - params['Z0'])
                 - params['OMEGA'] * t
                 + 1 / (2 * params['KZ'] * params['H'])
                 - phi))

def get_anal_ux(params, t, x, z, phi=0):
    uz_est = F * get_uz_f_ratio(params)
    k_damp = get_k_damp(params)

    return uz_est * (
        np.exp((z - params['Z0']) / 2 * params['H'])
        * np.exp(-k_damp * (z[0] - params['Z0'])) * (
            params['KZ'] / params['KX']
            * np.sin(params['KX'] * x
                     + params['KZ'] * (z - params['Z0'])
                     - params['OMEGA'] * t
                     + 1 / (2 * params['KZ'] * params['H'])
                     - phi)
            - np.cos(params['KX'] * x
                     + params['KZ'] * (z - params['Z0'])
                     - params['OMEGA'] * t
                     + 1 / (2 * params['KZ'] * params['H'])
                     - phi)
                / (2 * params['H'] * params['KX'])))

def get_times(time_fracs, sim_times, start_idx):
    return [int((len(sim_times) - start_idx) * time_frac + start_idx - 1)
            for time_frac in time_fracs]

def get_stats(arr):
    return np.median(arr), np.percentile(arr, 16), np.percentile(arr, 84)

def subtract_lins(params, state_vars, sim_times, domain):
    '''
    first subtract out reflected wave at each time, floating amp + phi
    then subtract out incident wave for time-independent amp, phi
    finally, subtract out retrograde wave at each time, floating amp + phi
    '''
    down_params = {**params, **{'KZ': -KZ}}

    x = domain.grid(0, scales=1)
    z = domain.grid(1, scales=1)
    x_t = np.array([x] * len(sim_times)) # all-time
    z_t = np.array([z] * len(sim_times))
    grid_ones = np.ones_like(x + z)
    t_t = np.array([[[t]] for t in sim_times])

    z_bot = get_idx(Z0 + 3 * S, z[0])
    z_top = get_idx(Z0 + 3 * S + 2 * Z_TOP_MULT * np.pi / abs(KZ), z[0])
    pos_slice = np.s_[:, z_bot:z_top]
    time_slice = np.s_[:, :, z_bot:z_top]

    dx = domain.grid_spacing(0, scales=1)
    dz = domain.grid_spacing(1, scales=1)
    dxdz = np.outer(dx, dz)[pos_slice]
    norm = np.exp(-z/H)[pos_slice]

    ux = state_vars['ux']
    uz = state_vars['uz']
    dphi_pixel = KX * XMAX / N_X * 2

    def obj_func(p, ux, uz, sim_time, params):
        ''' objective function for single-time minimization '''
        amp, offset = p
        resx = ux[pos_slice] - amp *\
            get_anal_ux(params, sim_time, x, z, phi=offset)[pos_slice]
        resz = uz[pos_slice] - amp *\
            get_anal_uz(params, sim_time, x, z, phi=offset)[pos_slice]
        return np.sum((resx**2 * KZ**2 + resz**2 * KZ**2 * norm) * dxdz)

    def obj_func_all_time(p, ux, uz, params):
        ''' objective function for all-time minimization '''
        amp, offset = p
        resx = ux[time_slice] - amp *\
            get_anal_ux(params, t_t, x_t, z_t, phi=offset)[time_slice]
        resz = uz[time_slice] - amp *\
            get_anal_uz(params, t_t, x_t, z_t, phi=offset)[time_slice]
        return np.sum((resx**2 * KZ**2 + resz**2 * KZ**2 * norm) * dxdz)

    def get_fits(fit_params, dux, duz):
        ''' for fit_params, fits dux, dux up to A, dphi at all times '''
        amps = []
        dphis = []
        duxs = []
        duzs = []
        for t_idx, sim_time in enumerate(sim_times):
            fit_anal_ux = get_anal_ux(fit_params, sim_time, x, z)[pos_slice]
            fit_anal_uz = get_anal_uz(fit_params, sim_time, x, z)[pos_slice]

            amp_est = 0
            off_est = 0
            # nearest-pixel search only for upwards
            for offset in range(N_X):
                curr = np.sum((
                    np.roll(dux[t_idx][pos_slice], -offset, axis=0)
                        * fit_anal_ux * KX**2 * norm +
                    np.roll(duz[t_idx][pos_slice], -offset, axis=0)
                        * fit_anal_uz * KZ**2 * norm
                ) * dxdz) / np.sum((
                    fit_anal_ux**2 * KX**2 * norm +
                    fit_anal_uz**2 * KZ**2 * norm) * dxdz)
                if curr > amp_est:
                    amp_est = curr
                    off_est = offset

            dphi_est = (off_est * KX * XMAX / N_X)
            fit = minimize(obj_func,
                           [amp_est, dphi_est],
                           (dux[t_idx], duz[t_idx], sim_time, fit_params),
                           bounds=[(0, 1.5),
                                   (dphi_est - dphi_pixel,
                                    dphi_est + dphi_pixel)])
            amp, dphi = fit.x
            amps.append(amp)
            dphis.append(dphi)
            print('Got fits', amp, dphi)

            duxs.append(
                dux[t_idx] -
                amp * get_anal_ux(fit_params, sim_time, x, z, phi=dphi))
            duzs.append(
                duz[t_idx] -
                amp * get_anal_uz(fit_params, sim_time, x, z, phi=dphi))
        return amps, dphis, np.array(duxs), np.array(duzs)

    # fit_incident = minimize(obj_func_all_time,
    #                         [1, 0],
    #                         (ux, uz, params),
    #                         bounds=[(0.5, 1.5), (-1, 1)])
    # amp, dphi = fit_incident.x
    # print(amp, dphi)
    # ux_inc_subbed = ux - amp * get_anal_ux(params, t_t, x_t, z_t, phi=dphi)
    # uz_inc_subbed = uz - amp * get_anal_uz(params, t_t, x_t, z_t, phi=dphi)

    # amps_down, dphis_down, ux_down_subbed, uz_down_subbed =\
    #     get_fits(down_params, ux_inc_subbed, uz_inc_subbed)

    # amps_retro, dphis_retro, duxs, duzs =\
    #     get_fits(params, ux_down_subbed, uz_down_subbed)

    amps, dphis, ux_inc_subbed, uz_inc_subbed =\
        get_fits(params, ux, uz)

    amps_down, dphis_down, duxs, duzs =\
        get_fits(down_params, ux_inc_subbed, uz_inc_subbed)

    return amps, dphis, amps_down, dphis_down, duxs, duzs

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
                         N_X,
                         interval=(0, XMAX),
                         dealias=3/2)
    z_basis = de.Chebyshev('z',
                           N_Z,
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

def run_strat_sim(name, params):
    populate_globals(params)
    snapshots_dir = SNAPSHOTS_DIR % name

    solver, domain = get_solver(params)

    # Initial conditions
    _, dt = set_ic(name, solver, domain, params)

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

def load(name, params, dyn_vars, stride, start=0):
    populate_globals(params)
    snapshots_dir = SNAPSHOTS_DIR % name
    merge(name)

    solver, domain = get_solver(params)
    z = domain.grid(1, scales=1)

    i = 1
    filename = FILENAME_EXPR.format(s=snapshots_dir, idx=i)
    total_sim_times = []
    state_vars = defaultdict(list)

    while os.path.exists(filename):
        print('Loading %s' % filename)
        with h5py.File(filename, mode='r') as dat:
            sim_times = np.array(dat['scales']['sim_time'])
            for varname in dyn_vars:
                state_vars[varname].extend(
                    dat['tasks'][varname][start::stride])

        simlen = len(sim_times)
        total_sim_times.extend(sim_times[start::stride])
        print('Loaded %d times' % len(sim_times[start::stride]))
        i += 1
        if simlen <= start:
            start -= simlen
        else:
            start = (((simlen - start - 1) // stride) + 1) * stride\
                + start - simlen
        filename = FILENAME_EXPR.format(s=snapshots_dir, idx=i)
    print('Loaded total %d times' % len(total_sim_times))

    # cast to np arrays
    for key in state_vars.keys():
        state_vars[key] = np.array(state_vars[key])

    if not NL:
        state_vars['ux_z'] = np.gradient(state_vars['ux'], axis=2)

    state_vars['S'] = RHO0 * np.exp(-z/ H) * (
        (state_vars['ux'] * state_vars['uz']) +
        (NU * np.exp(state_vars['U']) * state_vars['ux_z']))
    return np.array(total_sim_times), domain, state_vars

def plot(name, params, stride=STRIDE):
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
        # {
        #     'save_fmt_str': 'p_%03i.png',
        #     'plot_vars': ['ux', 'uz'],
        #     'res_vars': ['ux', 'uz'],
        # },
        {
            'save_fmt_str': 's_%03i.png',
            'slice_vars': ['uz', 'ux_z'],
            'mean_vars': ['S', 'ux'],
        },
        {
            'save_fmt_str': 'm_%03i.png',
            'plot_vars': ['uz', 'ux', 'W', 'S'],
        },
    ]

    dyn_vars = ['uz', 'ux', 'U', 'W']
    if NL:
        dyn_vars += ['ux_z']
    sim_times, domain, state_vars = load(name, params, dyn_vars, stride,
        start=0)

    x = domain.grid(0, scales=1)
    z = domain.grid(1, scales=1)
    z0 = z[0]
    xmesh, zmesh = quad_mesh(x=x[:, 0], y=z0)
    x2mesh, z2mesh = quad_mesh(x=np.arange(N_X // 2), y=z0)

    # preprocess
    for var in dyn_vars + ['S']:
        state_vars[var + mean_suffix] = horiz_mean(state_vars[var], N_X)

    for var in dyn_vars:
        state_vars[var + slice_suffix] = np.copy(state_vars[var][:, 0, :])

    for var in dyn_vars:
        # can't figure out how to numpy this together
        means = state_vars[var + mean_suffix]
        state_vars[var + sub_suffix] = np.copy(state_vars[var])
        for idx, _ in enumerate(state_vars[var + sub_suffix]):
            mean = means[idx]
            state_vars[var + sub_suffix][idx] -= np.tile(mean, (N_X, 1))
    _, _, _, _, dux2s, duz2s = \
        subtract_lins(params, state_vars, sim_times, domain)

    for cfg in plot_cfgs:
        n_cols, n_rows, save_fmt_str, plot_vars, f_vars,\
            f2_vars, mean_vars, slice_vars, sub_vars, res_vars\
            = get_plot_vars(cfg)

        uz_est = F * get_uz_f_ratio(params)
        ux_est = uz_est * KZ / KX

        for t_idx, time in list(enumerate(sim_times)):
            fig = plt.figure(dpi=DPI)

            uz_anal = get_anal_uz(params, time, x, z)
            ux_anal = get_anal_ux(params, time, x, z)
            uz_mean = np.outer(np.ones(N_X),
                               state_vars['uz%s' % mean_suffix][t_idx])
            ux_mean = np.outer(np.ones(N_X),
                               state_vars['ux%s' % mean_suffix][t_idx])
            S_px_mean = state_vars['S%s' % mean_suffix]
            z_top = get_front_idx(S_px_mean,
                                  t_idx,
                                  z0,
                                  get_flux_th(params) * 0.2)
            z_bot = get_idx(Z0 + 3 * S, z0)

            idx = 1
            for var in plot_vars + sub_vars + res_vars:
                if res_suffix in var:
                    # divide by analytical profile and normalize
                    dux2 = dux2s[t_idx]
                    duz2 = duz2s[t_idx]
                    if var == 'uz%s' % res_suffix:
                        var_dat = duz2 / (uz_est * np.exp((z - Z0) / (2 * H)))
                        title = 'u_z'
                    elif var == 'ux%s' % res_suffix:
                        var_dat = dux2 / (ux_est * np.exp((z - Z0) / (2 * H)))
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
                var_dat = state_vars[var]

                p = axes.plot(var_dat[t_idx],
                              z0,
                              'r-',
                              linewidth=0.7,
                              label='Data')
                if var == 'uz%s' % slice_suffix:
                    p = axes.plot(
                        uz_anal[0, :],
                        z0,
                        'orange',
                        linewidth=0.5)
                    p = axes.plot(
                        -uz_est * np.exp((z[0] - Z0) / (2 * H)), z[0], 'g',
                        uz_est * np.exp((z[0] - Z0) / (2 * H)), z[0], 'g',
                        linewidth=0.5)

                if var == 'ux%s' % slice_suffix:
                    p = axes.plot(
                        ux_anal[0, :],
                        z0,
                        'orange',
                        linewidth=0.5)
                    p = axes.plot(
                        -ux_est * np.exp((z0 - Z0) / (2 * H)), z[0], 'g',
                        ux_est * np.exp((z0 - Z0) / (2 * H)), z[0], 'g',
                        linewidth=0.5)

                if var == 'S%s' % mean_suffix:
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
                    Spx01 = horiz_mean(rho0 * state_vars['ux'][t_idx] *
                                       (state_vars['uz'][t_idx] - uz_anal),
                                       N_X, axis=0)
                    Spx10 = horiz_mean(rho0 * state_vars['uz'][t_idx] *
                                       (state_vars['ux'][t_idx] - ux_anal),
                                       N_X, axis=0)
                    Spx11 = horiz_mean(rho0 *
                                       (state_vars['ux'][t_idx] - ux_anal) *
                                       (state_vars['uz'][t_idx] - uz_anal),
                                       N_X, axis=0)
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
                            * np.exp((z0 - Z0) / H),
                        z[0],
                        'orange',
                        linewidth=0.5)
                    # critical = omega / kx
                    p = axes.plot(OMEGA / KX * np.ones_like(z0),
                        z0,
                        'green',
                        linewidth=0.5)

                plt.xticks(rotation=30)
                plt.yticks(rotation=30)
                xlims = [var_dat.min(), var_dat.max()]
                axes.set_xlim(*xlims)
                axes.set_ylim(z0.min(), z0.max())
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
                (name.replace('_', '.'), time, KX, KZ, OMEGA))
            fig.subplots_adjust(hspace=0.7, wspace=0.6)
            savefig = save_fmt_str % (t_idx)
            plt.savefig('%s/%s' % (snapshots_dir, savefig))
            logger.info('Saved %s/%s' % (snapshots_dir, savefig))
            plt.close()

def write_front(name, params, stride=1):
    ''' few plots for front, defined where flux drops below 1/2 of theory '''
    populate_globals(params)
    u_c = OMEGA / KX
    dyn_vars = ['uz', 'ux', 'U']
    if NL:
        dyn_vars += ['ux_z']
    snapshots_dir = SNAPSHOTS_DIR % name
    logfile = '%s/data.pkl' % snapshots_dir

    flux_th = get_flux_th(params)
    flux_threshold = flux_th * 0.3

    sim_times, domain, state_vars = load(
        name, params, dyn_vars, stride=stride, start=10)
    x = domain.grid(0, scales=1)
    z = domain.grid(1, scales=1)
    dz = domain.grid_spacing(1, scales=1)[0]
    z0 = z[0]
    rho0 = RHO0 * np.exp(-z0 / H)

    ri_med = []
    ri_min = []
    ri_max = []
    width_med = []
    width_min = []
    width_max = []
    S_top_ffts = []
    S_bot_ffts = []
    Spx11 = []

    S_px = horiz_mean(state_vars['S'], N_X)
    u0 = horiz_mean(state_vars['ux'], N_X)

    uz_est = F * get_uz_f_ratio(params)
    ux_est = uz_est * KZ / KX
    amps, phis, amps_down, phis_down, dux2s, duz2s =\
        subtract_lins(params, state_vars, sim_times, domain)
    for t_idx, sim_time in enumerate(sim_times):
        dux2 = dux2s[t_idx]
        duz2 = duz2s[t_idx]
        # heuristic: search near front_idx at each x
        front_idx = get_front_idx(S_px, t_idx, z0, flux_threshold)
        search_ux_z = state_vars['ux_z'][t_idx, :,
                                         front_idx - 10: front_idx + 10]
        ri_arr = (g / H) * np.min(1 / search_ux_z**2, axis=1)
        ri_med.append(np.median(ri_arr))
        ri_min.append(np.min(ri_arr))
        ri_max.append(np.max(ri_arr))

        width_arr = []
        for x_idx in range(N_X):
            ux_slice = state_vars['ux'][t_idx, x_idx]
            def interp_idx(f, idx, val):
                df_grid = f[idx] - f[idx - 1]
                dz_grid = z0[idx] - z0[idx - 1]
                df_val = val - f[idx - 1]
                if val > f[idx - 1] and val < f[idx]:
                    return z0[idx - 1] + dz_grid * df_val / df_grid
                return z0[idx]
            z_top_where = np.where(ux_slice[front_idx - 5: ] > u_c)[0]
            z_bot_where = np.where(
                ux_slice[ : front_idx + 5] < 0.3 * u_c)[0]

            if not len(z_top_where) or not len(z_bot_where):
                width_arr.append(ZMAX)
                continue
            z_top = interp_idx(ux_slice, z_top_where[0] + front_idx - 5, u_c)
            z_bot = interp_idx(ux_slice, z_bot_where[-1] + 1, 0.3 * u_c)
            width_arr.append(z_top - z_bot)
        width_med.append(np.median(width_arr))
        width_min.append(np.min(width_arr))
        width_max.append(np.max(width_arr))

        window_width = 2 * Z_TOP_MULT * np.pi / abs(KZ)

        z_bot_l = get_idx(z0[front_idx] - 2 * window_width, z0)
        z_bot_r = get_idx(z0[front_idx] - 1 * window_width, z0)
        area_bot = np.outer(np.ones_like(z[:, 0]), dz[z_bot_l: z_bot_r])
        S_bot = state_vars['S'][t_idx, :, z_bot_l: z_bot_r]
        S_bot_fft = np.abs(np.fft.rfft(S_bot / flux_th, axis=0) / N_X)
        # by using only half of the fft, all non-DC bins are half as high as
        # they should be
        S_bot_fft[1: ] *= 2
        # area_bot isn't really the area element, since this is (kx, z) space,
        # but kx, x are both evenly distributed so it's ok
        S_bot_ffts.append(np.sum(S_bot_fft * area_bot, axis=1) /
                          np.sum(area_bot, axis=1))

        z_top_r = get_idx(z0[front_idx] + 2 * window_width, z0)
        z_top_l = get_idx(z0[front_idx] + 1 * window_width, z0)
        area_top = np.outer(np.ones_like(z[:, 0]), dz[z_top_l: z_top_r])
        S_top = state_vars['S'][t_idx, :, z_top_l: z_top_r]
        S_top_fft = np.abs(np.fft.rfft(S_top / flux_th, axis=0) / N_X)
        S_top_fft[1: ] *= 2
        S_top_ffts.append(np.sum(S_top_fft * area_top, axis=1) /
                          np.sum(area_top, axis=1))

        Spx11.append(horiz_mean(rho0 * dux2 * duz2, N_X, axis=0))

    with open(logfile, 'wb') as data:
        pickle.dump((
            z0, sim_times, S_px, Spx11, u0,
            np.array(ri_med), np.array(ri_min), np.array(ri_max),
            np.array(width_med), np.array(width_min), np.array(width_max),
            np.array(amps), np.array(phis),
            np.array(amps_down), np.array(phis_down),
            np.array(S_bot_ffts), np.array(S_top_ffts),
        ), data)

def get_dS_front(params, S_px, sim_times, z0):
    populate_globals(params)
    flux_th = get_flux_th(params)
    flux_threshold = flux_th * 0.3
    dSpx = [] # Delta S_px
    front_pos = []
    front_idxs = []
    dz = abs(3 / KZ)
    l_z = abs(2 * np.pi / KZ)
    z_b = Z0 + 3 * S
    z_b_idx = get_idx(z_b, z0)
    S_aboves = []
    for t_idx, sim_time in enumerate(sim_times):
        front_idx = get_front_idx(S_px, t_idx, z0, flux_threshold)
        z_c = z0[front_idx]
        front_pos.append(z_c)
        front_idxs.append(front_idx)

        # measure flux incident at dz below critical layer, small time/space avg
        z_bot_idx = get_idx(z_c - dz, z0)
        z_top_idx = get_idx(z_c + dz, z0)
        S_below = np.mean(S_px[max(t_idx - AVG_IDX, 0): t_idx + AVG_IDX,
                               z_bot_idx])
        S_above_crit = np.mean(S_px[max(t_idx - AVG_IDX, 0): t_idx + AVG_IDX,
                                    front_idx:z_top_idx], axis=0)
        S_above = np.mean(S_above_crit[np.where(S_above_crit < 0)[0]])
        if np.isnan(S_above):
            S_above = 0
        # S_below = S_px[t_idx, z_bot_idx]
        # S_above = S_px[t_idx, z_top_idx]
        dSpx.append(S_above - S_below)
        S_aboves.append(S_above)
    return np.array(dSpx), np.array(front_pos), front_idxs, np.array(S_aboves)

def plot_front(name, params):
    populate_globals(params)
    N = np.sqrt(g / H)
    u_c = OMEGA / KX
    V_PZ = abs(get_vpz(g, H, KX, KZ))
    V_GZ = abs(get_vgz(g, H, KX, KZ))
    flux_th = get_flux_th(params)
    k_damp = get_k_damp(params)

    dyn_vars = ['uz', 'ux', 'U']
    snapshots_dir = SNAPSHOTS_DIR % name
    logfile = '%s/data.pkl' % snapshots_dir
    if NL:
        dyn_vars += ['ux_z']
    if not os.path.exists(logfile):
        print('Generating logfile')
        write_front(name, params)
    else:
        print('Loading data')

    with open(logfile, 'rb') as data:
        z0, sim_times, S_px, Spx11, u0, ri_med, ri_min, ri_max,\
            width_med, width_min, width_max,\
            amps, phis, amps_down, phis_down, \
            S_bot_ffts, S_top_ffts = pickle.load(data)
        Spx11 = np.array(Spx11) / flux_th

    tf = sim_times[-1]
    start_idx = get_idx(200, sim_times)
    t = sim_times[start_idx: ]

    S_px0 = amps**2 * flux_th
    dz = abs(3 / KZ)
    l_z = abs(2 * np.pi / KZ)
    z_b = Z0 + 3 * S
    z_b_idx = get_idx(z_b, z0)
    dSpx, front_pos, front_idxs, S_aboves =\
        get_dS_front(params, S_px, sim_times, z0)

    times = get_times([1/9, 1/6, 1/3, 1], sim_times, start_idx)
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
                     label=r'$t=%.1f$' % sim_times[time])
        # plt.plot(z0_cut,
        #          np.exp(-k_damp * 2 * (z0_cut - Z0)),
        #          linewidth=1.5,
        #          label=r'Model')
        plt.xlim(z_b, ZMAX)
        plt.ylim(-0.1, 1.1)
        plt.legend(fontsize=FONTSIZE)

        plt.xlabel(r'$z(H)$')
        plt.ylabel(r'$\hat{F}$')
        plt.savefig('%s/fluxes.png' % snapshots_dir, dpi=DPI)
        plt.close()

        #####################################################################
        # f_amps.png
        #
        # convolved amplitudes over time
        #####################################################################
        f, ax1 = plt.subplots(1, 1, sharex=True)
        ax1.plot(t,
                 smooth(amps[start_idx: ]),
                 'g',
                 label=r'$A_i$',
                 linewidth=1.0)
        # ax1.plot(t,
        #          smooth(amps_down[start_idx: ]),
        #          'r',
        #          label=r'$A_d$',
        #          linewidth=0.7)
        # ax1.legend(fontsize=FONTSIZE)
        ax1.set_xlabel(r'$t$')
        ax1.set_ylabel(r'$\hat{A}$')
        # ax2.plot(t,
        #          np.unwrap(phis_down[start_idx: ]),
        #          'r',
        #          label=r'$\phi_d$',
        #          linewidth=0.7)
        # ax2.plot(t,
        #          np.unwrap(phis[start_idx: ]),
        #          'k',
        #          label=r'$\phi_I$',
        #          linewidth=0.7)
        plt.savefig('%s/f_amps.png' % snapshots_dir, dpi=DPI)
        plt.close()
        avg_refl, avg_reflA, avg_ri, avg_trans = tuple([tuple([0, 0, 0])] * 4)

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
            u0_avg = np.mean(u0[time - AVG_IDX: time + AVG_IDX,
                                z_b_idx: ], axis=0)
            S_px_avg = np.mean(S_px[time - AVG_IDX: time + AVG_IDX,
                                    z_b_idx: ], axis=0)
            # time-averaged mean flow
            ax1.plot(z0_cut,
                     u0_avg / u_c,
                     '%s-' % color,
                     linewidth=0.5,
                     label=r'$t=%.1f$' % sim_times[time])
            # S_px sliced at time, just one for comparison
            # if time == times[len(times) // 2]:
            #     ax2.plot(z0_cut,
            #              S_px[time, z_b_idx: ] / flux_th,
            #              '%s:' % color,
            #              linewidth=0.4)
            # S_px time-averaged
            ax2.plot(z0_cut,
                     S_px_avg / flux_th,
                     '%s-' % color,
                     linewidth=0.5,
                     label=r'$t=%.1f$' % sim_times[time])
        # overlay analytical flux including viscous dissipation
        # ax2.plot(z0_cut,
        #          np.exp(-k_damp * 2 * (z0_cut - Z0)),
        #          linewidth=1.5,
        #          label=r'$\nu$-only')
        ax1.set_xlim(z_b, ZMAX)
        ax2.set_xlim(z_b, ZMAX)
        ax1.set_ylim(-0.2, 1.25)
        ax2.set_ylim(-0.2, 1.2)
        ax2.legend(fontsize=FONTSIZE)

        ax1.set_ylabel(r'$\bar{U} k_x/\omega$')
        ax2.set_ylabel(r'$\hat{F}$')
        ax2.set_xlabel(r'$z(H)$')
        plt.savefig('%s/fluxes.png' % snapshots_dir, dpi=DPI)
        plt.close()

        #####################################################################
        # front.png
        #
        # plot front position and absorbed flux over time
        #####################################################################
        f, (ax2) = plt.subplots(1, 1, sharex=True)
        f.subplots_adjust(hspace=0)

        zf = np.max(front_pos[-5: -1])

        # compute front position
        front_vel_S = dSpx / (RHO0 * np.exp(-front_pos / H) * u_c)
        front_pos_intg_S = np.cumsum((front_vel_S * np.gradient(sim_times))
                                      [start_idx: ])
        front_pos_intg_S += zf - front_pos_intg_S[-1]

        # estimate incident Delta S_px if all from S_px0 that's viscously
        # damped, compare to other Delta S_px criteria/from data
        # color_idx = 0
        # dSpx0 = S_px0[start_idx: ] / flux_th * \
        #     np.exp(-k_damp * 2 * (front_pos[start_idx: ] - (z_b + l_z / 2)))
        # ax1.plot(t,
        #          smooth(S_px0[start_idx: ] / flux_th),
        #          '%s-' % PLT_COLORS[color_idx],
        #          label=r'$F(z=z_0)$',
        #          linewidth=0.7)
        # color_idx += 1
        # ax1.plot(t,
        #          smooth(dSpx0),
        #          '%s-' % PLT_COLORS[color_idx],
        #          label=r'$\Delta F_{1}|_{z=z_{c}}$',
        #          linewidth=0.7)
        # color_idx += 1
        # ax1.plot(t,
        #          smooth(-dSpx[start_idx: ] / flux_th),
        #          '%s-' % PLT_COLORS[color_idx],
        #          label=r'$\Delta F(z_{c})$',
        #          linewidth=0.7)
        # color_idx += 1
        # ax1.set_ylabel(r'$F / F_1$')
        # ax1.legend(fontsize=FONTSIZE, loc='lower right')

        # compare forecasts of front position using two predictors integrated
        # from incident flux in data
        ax2.plot(t,
                 front_pos[start_idx: ],
                 'k-',
                 label='Data',
                 linewidth=0.7)
        ax2.plot(t,
                 front_pos_intg_S,
                 'g-',
                 label='Model',
                 linewidth=0.7)

        # estimate front position using just average absorbed flux
        mean_incident = -np.mean(dSpx)
        est_incident_flux = np.mean(S_px0 *
                                    np.exp(-k_damp * 2 * (front_pos - Z0)))
        tau = H * RHO0 * u_c / mean_incident
        pos_anal = -H * np.log(
            (t - tf + tau * np.exp(-zf/H))
            / tau)
        ax2.plot(t,
                 pos_anal,
                 'b:',
                 label="Avg Absorbed $%.2fF_{al}'$" %
                    (mean_incident / flux_th),
                 linewidth=0.7)
        ax2.set_ylabel(r'$z_c$')
        ax2.set_xlabel(r'$t$')
        ax2.set_ylim([zf, np.max(front_pos[start_idx: ])])
        ax2.legend(fontsize=FONTSIZE, loc='upper right')
        plt.savefig('%s/front.png' % snapshots_dir, dpi=DPI)
        plt.close()

        #####################################################################
        # f_amps.png
        #
        # convolved amplitudes over time
        #####################################################################
        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        f.subplots_adjust(hspace=0)
        ax1.plot(t,
                smooth(amps[start_idx::]),
                'g',
                label=r'$\hat{A}_i(t)$',
                linewidth=1.0)
        ax1.plot(t,
                 smooth(amps_down[start_idx::]),
                 'r',
                 label=r'$\hat{A}_r(t)$',
                 linewidth=0.7)
        ax1.set_ylabel(r'$\hat{A}$')
        ax1.set_ylim(bottom=0)
        ax1.legend(loc=0, fontsize=FONTSIZE)

        # ax2.plot(t,
        #          np.unwrap(phis_down[start_idx: ]),
        #          'r',
        #          label=r'$\phi_d$',
        #          linewidth=0.7)
        # ax2.plot(t,
        #          np.unwrap(phis[start_idx: ]),
        #          'k',
        #          label=r'$\phi_I$',
        #          linewidth=0.7)
        # ax2.set_ylabel(r'$\phi$')

        S_excited = S_px0[start_idx: ] / flux_th * \
            np.exp(-k_damp * 2 * (front_pos[start_idx: ] - (z_b + l_z / 2)))
        ax2.plot(t,
                 smooth(S_excited),
                 'b:',
                 label=r'$F_i(t)$',
                 linewidth=0.7)
        ax2.plot(t,
                 -smooth(dSpx[start_idx: ]) / flux_th,
                 'r:',
                 label=r'$\Delta F(t)$',
                 linewidth=1.0)
        ax2.plot(t,
                 smooth(S_aboves[start_idx: ]) / flux_th,
                 'k:',
                 label=r'$F_>(t)$',
                 linewidth=1.0)
        ax2.set_xlabel(r'$t$')
        ax2.set_ylabel(r"$F / F_{al}'$")
        ax2.legend(fontsize=FONTSIZE, loc='lower left')
        plt.savefig('%s/f_amps.png' % snapshots_dir, dpi=DPI)
        plt.close()

        #####################################################################
        # f_refl.png
        #
        # reflection coeff calculations
        #####################################################################

        def my_interp(t, f):
            # seems like sometimes t has duplicate entries
            idxs = np.where(np.diff(t) > 0)
            f_smooth = smooth(np.concatenate((f[idxs], [f[-1]])))
            return interp1d(np.concatenate((t[idxs], [t[-1]])), f_smooth)

        # prop_time = (smooth(front_pos[start_idx: ]) -
        #              (Z0 + 3 * S + Z_TOP_MULT * np.pi / abs(KZ)))\
        #              / abs(V_PZ)
        prop_time = 0 * t

        t_refl = np.linspace((t + prop_time)[0], (t - prop_time)[-1], len(t))
        f, ax1 = plt.subplots(1, 1, sharex=True)

        absorbed_dS = my_interp(t, -(dSpx - S_aboves)[start_idx: ] / flux_th)
        trans_dS = my_interp(t, -S_aboves[start_idx: ] / flux_th)
        incident_dS = my_interp(t + prop_time, S_excited)
        refl = [(incident_dS(t) - absorbed_dS(t)) / incident_dS(t)
                for t in t_refl]
        trans = [trans_dS(t) / incident_dS(t) for t in t_refl]

        amps_interp = my_interp(t + prop_time, amps[start_idx: ])
        amps_down_interp = my_interp(t - prop_time, amps_down[start_idx: ])
        refl_amp = np.array([amps_down_interp(t) / amps_interp(t)
                             for t in t_refl]) * \
            np.exp(+k_damp * (front_pos[start_idx: ] - (z_b + l_z / 2)))

        avg_refl = get_stats(refl[int(len(refl) * 3 / 4): ])
        avg_reflA = get_stats(refl_amp[int(len(refl_amp) * 3 / 4): ])
        avg_trans = get_stats(trans[int(len(trans) * 3 / 4): ])

        ax1.plot(t_refl, refl, 'r:', linewidth=0.7,
                 label=r'$\mathcal{R}_F$')
        ax1.plot(t_refl, refl_amp**2, 'g:', linewidth=0.7,
                 label='$\mathcal{R}_A^2$')
        ax1.plot(t_refl, trans, 'k:', linewidth=0.7, label='$\mathcal{T}_F$')

        ax1.legend(fontsize=FONTSIZE, loc='upper left')
        ax1.set_ylabel(r'Reflectivity')
        ax1.set_xlabel(r'$t$')
        ax1.set_ylim([0, 0.5])

        plt.savefig('%s/f_refl.png' % snapshots_dir, dpi=DPI)
        plt.close()

        #####################################################################
        # f_ri.png
        #
        # richardson number plot
        #####################################################################

        f, ax1 = plt.subplots(1, 1, sharex=True)
        ri_width = N**2 * width_med**2 / (0.7 * u_c)**2
        ax1.plot(t, ri_width[start_idx: ], 'g', linewidth=0.7, label='Med')
        ri_min = N**2 * width_min**2 / (0.7 * u_c)**2
        ax1.plot(t, ri_min[start_idx: ], 'r:', linewidth=0.7, label='Min')
        # ri_max = N**2 * width_max**2 / (0.7 * u_c)**2
        # ax1.plot(t, ri_max[start_idx: ], 'r:', linewidth=0.7, label='Max')
        ax1.set_ylim([0, 0.6])
        ax1.set_ylabel(r"Ri $(N / \bar{U}')^2$")
        ax1.set_xlabel(r'$t$')
        avg_ri = get_stats(ri_width[len(ri_width) // 2: ])
        plt.legend(loc='lower left', fontsize=FONTSIZE)
        plt.savefig('%s/f_ri.png' % snapshots_dir, dpi=DPI)
        plt.close()

    #########################################################################
    # fft.png
    #
    # plot FFTs of residuals
    #########################################################################
    # f, ax1 = plt.subplots(1, 1, sharex=True)
    # f.subplots_adjust(hspace=0)
    # num_modes = 20
    # uz_est = F * get_uz_f_ratio(params)
    # ux_est = uz_est * KZ / KX

    # times = get_times([1/4, 1/2, 4/5], sim_times, start_idx)
    # smoothed_bot = np.array([smooth(t_slice)
    #                          for t_slice in (S_bot_ffts[:, : num_modes]).T]).T
    # smoothed_top = np.array([smooth(t_slice)
    #                          for t_slice in (S_top_ffts[:, : num_modes]).T]).T
    # for t_idx, color in zip(times, PLT_COLORS):
    #     ax1.plot(smoothed_bot[t_idx],
    #                color,
    #                label='t=%.1f' % sim_times[t_idx],
    #                linewidth=0.7)
    #     ax1.plot(smoothed_top[t_idx],
    #              '%s:' % color,
    #                linewidth=0.5)
    # ax1.set_ylabel(r'$\tilde{F}$')
    # ax1.set_xlabel(r'$k_x/k_{x1}$')
    # ax1.legend(fontsize=FONTSIZE)
    # plt.savefig('%s/fft.png' % snapshots_dir, dpi=DPI)
    # plt.close()

    # return aggregated values
    return avg_refl, avg_reflA, avg_trans, avg_ri
