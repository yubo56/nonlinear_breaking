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
from scipy.optimize import minimize

from dedalus import public as de
from dedalus.extras.flow_tools import CFL, GlobalFlowProperty

SNAPSHOTS_DIR = 'snapshots_%s'
FILENAME_EXPR = '{s}/{s}_s{idx}.h5'
plot_stride = 15

def get_uz_f_ratio(S, KX, KZ, OMEGA, RHO0, Z0, H):
    ''' get uz(z = z0) / F '''
    return (np.sqrt(2 * np.pi) * S * g *
            KX**2) * np.exp(-S**2 * KZ**2/2) / (
                2 * RHO0 * np.exp(-Z0 / H)
                * OMEGA**2 * KZ)

if __name__ == '__main__':
    H = 1
    XMAX = 4 * H
    ZMAX = 10 * H
    NUM_SNAPSHOTS = 300
    TARGET_DISP_RAT = 0.1 # k_z * u_z / omega at base

    N_X = 256
    N_Z = 1024

    KX = 2 * np.pi / XMAX
    KZ = -2 * np.pi / H
    g = H
    N = np.sqrt(g / H)
    RHO0 = 1
    Z0 = 0.2 * ZMAX
    Re_inv = 1

    F_MULT = 1
    SPONGE_STRENGTH = 15
    SPONGE_WIDTH = 0.5
    SPONGE_HIGH = 0.95 * ZMAX
    SPONGE_LOW = 0.03 * ZMAX

    OMEGA = np.sqrt(N**2 * KX**2 / (KX**2 + KZ**2 + 0.25 / H**2))
    VG_Z = -OMEGA * KX * KZ / (KX**2 + KZ**2 + 0.25 / H**2)**(3/2)
    T_F = ZMAX / VG_Z * 2
    S = ZMAX / 128
    DT = 0.1

    F = (TARGET_DISP_RAT * OMEGA / KZ) \
        / get_uz_f_ratio(S, KX, KZ, OMEGA, RHO0, Z0, H) \
        * np.exp(-Z0 / (2 * H))
    NU = 0.3 *  OMEGA * ZMAX / (2 * np.pi * N_Z) / abs(KZ)
    snapshots_dir = SNAPSHOTS_DIR % 'yubo'

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

    variables = ['W', 'U', 'ux', 'uz', 'uz_z', 'ux_z', 'U_z']
    problem = de.IVP(domain, variables=variables)

    problem.parameters['H'] = H
    problem.parameters['ZMAX'] = ZMAX
    problem.parameters['KX'] = KX
    problem.parameters['g'] = g
    problem.parameters['RHO0'] = RHO0
    problem.parameters['Z0'] = Z0
    problem.parameters['S'] = S
    problem.parameters['SPONGE_STRENGTH'] = SPONGE_STRENGTH
    problem.parameters['SPONGE_WIDTH'] = SPONGE_WIDTH
    problem.parameters['SPONGE_HIGH'] = SPONGE_HIGH
    problem.parameters['SPONGE_LOW'] = SPONGE_LOW
    problem.parameters['OMEGA'] = OMEGA
    problem.parameters['T_F'] = T_F
    problem.parameters['F'] = F
    problem.parameters['NU'] = NU

    problem.substitutions['sponge'] = 'SPONGE_STRENGTH * 0.5 * ' +\
        '(2 + tanh((z - SPONGE_HIGH) / (SPONGE_WIDTH * (ZMAX - SPONGE_HIGH)))'+\
        '- tanh((z - SPONGE_LOW) / (SPONGE_WIDTH * (SPONGE_LOW))))'
    problem.substitutions['rho0'] = 'RHO0 * exp(-z / H)'
    problem.substitutions['NL_MASK'] = \
        '0.5 * (1 + tanh((z - (Z0 + 4 * S)) / S))'

    # equations
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

    # Build solver
    solver = problem.build_solver(de.timesteppers.RK443)
    solver.stop_sim_time = T_F
    solver.stop_wall_time = np.inf
    solver.stop_iteration = np.inf

    # Initial conditions
    if os.path.exists(snapshots_dir):
        filename = FILENAME_EXPR.format(s=snapshots_dir, idx=1)
        # try to load snapshots_s1
        print('Attempting to load snapshots in', filename)
        _, dt = solver.load_state(filename, -1)
        print('Loaded snapshots')
    else:
        print('No snapshots found')
        dt = DT

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
        cfl_dt = cfl.compute_dt()
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
