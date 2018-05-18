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

###
### UTILS
###

def get_omega(g, h, kx, kz):
    return np.sqrt((g / h) * kx**2 / (kx**2 + kz**2 + 0.25 / h**2))

def get_vph(g, h, kx, kz):
    norm = get_omega(g, h, kx, kz) / (kx**2 + kz**2)
    return norm * kx, norm * kz

def get_sponge(domain, params):
    sponge_strength = params['SPONGE_STRENGTH']
    zmax = params['ZMAX']
    damp_start = params['SPONGE_START']
    z = domain.grid(1)

    # sponge field
    sponge = domain.new_field()
    sponge.meta['x']['constant'] = True
    sponge['g'] = sponge_strength * (
        np.maximum(z - damp_start, 0) ** 2 / (1 - damp_start) ** 2 +
        np.maximum(0.2 - z, 0) ** 2 / (0.2) ** 2)
    return sponge

###
### IC
###

def zero_ic(solver, domain, params):
    ux = solver.state['ux']
    uz = solver.state['uz']
    P = solver.state['P']
    rho = solver.state['rho']
    gshape = domain.dist.grid_layout.global_shape(scales=1)

    P['g'] = np.zeros(gshape)
    ux['g'] = np.zeros(gshape)
    uz['g'] = np.zeros(gshape)
    rho['g'] = np.zeros(gshape)

###
### PROBLEM SETUP
###

def setup_problem(problem, domain, params):
    ''' sponge zone velocities w nonlin terms '''
    problem.parameters['sponge'] = get_sponge(domain, params)
    problem.add_equation('dx(ux) + dz(uz) = 0')
    problem.add_equation(
        'dt(rho) + sponge * rho - rho0 * uz / H' +
        '= -ux * dx(rho) - uz * dz(rho)'
        # '= 0'
    )
    problem.add_equation(
        'dt(ux) + sponge * ux + dx(P) / rho0' +
        '= - ux * dx(ux) - uz * dz(ux)'
        # '= 0'
    )
    problem.add_equation(
        'dt(uz) + sponge * uz + dz(P) / rho0 + rho * g / rho0' +
        '= - ux * dx(uz) - uz * dz(uz) +' +
        # '='
        '-F * (z - 0.1 * ZMAX) / (0.1**2) * ' +
            'exp(-(z - 0.1 * ZMAX)**2 / (2 * 0.1**2)) *' +
            'sin(KX * x - omega * t)')

    z = domain.grid(1)
    x = domain.grid(0)
    problem.add_bc('left(uz) = 0', condition='nx != 0')
    problem.add_bc('left(P) = 0', condition='nx == 0')
    problem.add_bc('right(uz) = 0')

###
### SOLVER SETUP
###

def _get_solver(setup_problem, params, variables):
    ''' get solver for given variables '''
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

    problem = de.IVP(domain, variables=variables)
    problem.parameters['L'] = params['XMAX']
    problem.parameters['g'] = params['G']
    problem.parameters['H'] = params['H']
    problem.parameters['A'] = params['A']
    problem.parameters['F'] = params['F']
    problem.parameters['KX'] = params['KX']
    problem.parameters['KZ'] = params['KZ']
    problem.parameters['NU'] = params['NU']
    problem.parameters['RHO0'] = params['RHO0']
    problem.parameters['omega'] = params['OMEGA']
    problem.parameters['ZMAX'] = params['ZMAX']

    # rho0 stratification
    rho0 = domain.new_field()
    rho0.meta['x']['constant'] = True
    rho0['g'] = params['RHO0'] * np.exp(-z / params['H'])
    problem.parameters['rho0'] = rho0

    setup_problem(problem, domain, params)

    # Build solver
    solver = problem.build_solver(de.timesteppers.RK443)
    solver.stop_sim_time = params['T_F']
    solver.stop_wall_time = np.inf
    solver.stop_iteration = np.inf
    return solver, domain

def get_solver(setup_problem, params):
    return _get_solver(setup_problem,
                       params,
                       variables=['P', 'rho', 'ux', 'uz'])

###
### ENTRY POINTS
###

def run_strat_sim(get_solver, setup_problem, set_ICs, name, params):
    snapshots_dir = SNAPSHOTS_DIR % name
    try:
        os.makedirs(snapshots_dir)
    except FileExistsError:
        print('snapshots already exist, exiting...')
        return
    logger = logging.getLogger(name)

    solver, domain = get_solver(setup_problem, params)

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

def load(get_solver, setup_problem, name, params):
    dyn_vars = ['uz', 'ux', 'rho', 'P']
    snapshots_dir = SNAPSHOTS_DIR % name
    filename = '{s}/{s}_s1/{s}_s1_p0.h5'.format(s=snapshots_dir)

    if not os.path.exists(snapshots_dir):
        raise ValueError('No snapshots dir "%s" found!' % snapshots_dir)

    solver, domain = get_solver(setup_problem, params)
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
        params['G'] * params['H']
    state_vars['rho1'] = state_vars['rho'] - params['RHO0'] *\
        np.exp(-z / params['H'])
    state_vars['P1'] = state_vars['P'] -\
        params['RHO0'] * (np.exp(-z / params['H']) - 1) *\
        params['G'] * params['H']

    state_vars['E'] = state_vars['rho'] * \
                       (state_vars['ux']**2 + state_vars['uz']**2) / 2
    state_vars['F_z'] = state_vars['uz'] * (
        state_vars['rho'] * (state_vars['ux']**2 + state_vars['uz']**2)
        + state_vars['P'])
    return sim_times, domain, state_vars

def plot(get_solver, setup_problem, name, params):
    slice_suffix = '(x=0)' # slice suffix
    SAVE_FMT_STR = 't_%d.png'
    snapshots_dir = SNAPSHOTS_DIR % name
    path = '{s}/{s}_s1'.format(s=snapshots_dir)
    matplotlib.rcParams.update({'font.size': 6})
    plot_vars = ['uz', 'ux']
    z_vars = ['F_z', 'E'] # sum these over x
    slice_vars = ['%s%s' % (i, slice_suffix)
                  for i in ['uz', 'ux', 'rho1', 'P1']]
    n_cols = 3
    n_rows = 3
    plot_stride = 2

    if os.path.exists('%s.mp4' % name):
        print('%s.mp4 already exists, not regenerating' % name)
        return

    sim_times, domain, state_vars = load(get_solver, setup_problem, name, params)

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

            plt.xticks(rotation=30)
            plt.yticks(rotation=30)
            xlims = [var_dat.min(), var_dat.max()]
            axes.set_xlim(*xlims)
            p = axes.plot(xlims, [params['SPONGE_START']] * len(xlims), 'r--')
            idx += 1

        fig.suptitle(
            'Config: %s (t=%.2f, kx=%.2f, kz=%.2f, omega=%.2f)' %
            (name, sim_time, params['KX'], params['KZ'], params['OMEGA']))
        fig.subplots_adjust(hspace=0.5, wspace=0.6)
        savefig = SAVE_FMT_STR % (t_idx // plot_stride)
        plt.savefig('%s/%s' % (path, savefig))
        print('Saved %s/%s' % (path, savefig))
        plt.close()
    os.system('ffmpeg -y -framerate 12 -i %s/%s %s.mp4' %
              (path, SAVE_FMT_STR, name))
