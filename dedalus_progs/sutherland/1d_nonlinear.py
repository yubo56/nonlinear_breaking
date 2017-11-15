"""
Eq. 26 from Sutherland et al 2011
"""

import logging
import numpy as np
import matplotlib.pyplot as plt

from dedalus import public as de
from dedalus.extras.plot_tools import quad_mesh, pad_limits

logger = logging.getLogger(__name__)
K = 1
N = 1
# ZMIN = -150 / K
# ZMAX = 150 / K
ZMIN = -20 / K
ZMAX = 80 / K
N_Z = 1024
DX = (ZMAX - ZMIN) / N_Z
# DT = 1e-4 / N
DT = 2e-4 / N

m = -0.4 * K
w = 0.93 * N
C_G = 0.32 * N / K
w_mm = -0.47 * N / K ** 2
w_mmm = -1.91 * N / K ** 3
H = 10 / K
sigm = 10 / K
A = 0.05 / K

T_F = 150 / N

try:
    # Bases and domain
    z_basis = de.Fourier('z', N_Z, interval=(ZMIN, ZMAX), dealias=1)
    domain = de.Domain([z_basis], np.float64)

    # Problem
    problem = de.IVP(domain, variables=['Ar',  'Ai'])
    problem.parameters['C_G'] = C_G
    problem.parameters['w_mm'] = w_mm
    problem.parameters['w_mmm'] = w_mmm
    problem.parameters['K'] = K
    problem.parameters['U0'] = (K ** 2 + m ** 2 + H ** 2 / 4) ** (3 / 2)\
            / (2 * N)
    problem.parameters['w'] = w
    problem.parameters['N'] = N
    problem.parameters['H'] = H
    problem.parameters['m'] = m
    problem.parameters['e'] = np.e
    # notes: only second term produces divergences...
    problem.add_equation('dt(Ar) + C_G * dz(Ar) = -\
        1 * w_mm * dz(dz(Ai)) / 2 +\
        0 * w_mmm * dz(dz(dz(Ar))) / 6 -\
        1 * K * U0 * (Ar**2 + Ai**2) * e**(z/H) * Ar +\
        1 * w**2 / (2 * N**2 * K * H) * (\
            (U0 * e**(z/H) * (2 * Ar * dz(Ar) + 2 * Ai * dz(Ai))) \
            + (U0 * (Ar ** 2 + Ai ** 2) * e**(z/H) / H)) *\
            (3 * m * H * Ar + Ai)')
    problem.add_equation('dt(Ai) + C_G * dz(Ai) =\
        1 * w_mm * dz(dz(Ar)) / 2 +\
        0 * w_mmm * dz(dz(dz(Ai))) / 6 +\
        1 * K * U0 * (Ar**2 + Ai**2) * e**(z/H) * Ai +\
        1 * w**2 / (2 * N**2 * K * H) * (\
            (U0 * e**(z/H) * (2 * Ar * dz(Ar) + 2 * Ai * dz(Ai))) \
            + U0 * (Ar ** 2 + Ai ** 2) * e**(z/H) / H) *\
            (3 * m * H * Ai - Ar)')

    # Build solver
    solver = problem.build_solver(de.timesteppers.SBDF2)
    solver.stop_sim_time = T_F
    solver.stop_wall_time = 10000000 # should never get hit
    solver.stop_iteration = 10000000 # should never get hit

    # Initial conditions
    z = domain.grid(0)
    zero_idx = -ZMIN / DX
    Ar = solver.state['Ar']
    Ai = solver.state['Ai']

    num_els = len(Ar['g'])
    Ar['g'] = 1e-20 * np.ones(num_els)
    Ai['g'] = 1e-20 * np.ones(num_els)
    for i in range(num_els):
        Ar['g'][i] += A * \
            np.exp(-min((zero_idx - i)**2, (zero_idx + num_els - i)**2)
            / (2 * (sigm / DX)**2))

    # Store data for final plot
    Ar.set_scales(1, keep_data=True)
    Ai.set_scales(1, keep_data=True)
    ar_list = [np.copy(Ar['g'])]
    ai_list = [np.copy(Ai['g'])]
    t_list = [solver.sim_time]

    # Main loop
    while solver.ok:
        solver.step(DT)
        if solver.iteration % round((T_F // DT) / 200) == 0:
            Ar.set_scales(1, keep_data=True)
            Ai.set_scales(1, keep_data=True)
            print('Maxes:', max(Ar['g']), max(Ai['g']))
            ar_list.append(np.copy(Ar['g']))
            ai_list.append(np.copy(Ai['g']))
            t_list.append(solver.sim_time)
            logger.info(
                'Iteration: %i, Time: %.3f/%.3f, dt: %.3e',
                solver.iteration,
                solver.sim_time,
                T_F,
                DT
            )

except Exception as e:
    print('Caught exception:', e, '\n\tTrying to plot')
except KeyboardInterrupt as e:
    print('Caught keyboard interrupt, plotting...')

if ar_list and ai_list:
    # Create space-time plot
    ar_arr = np.array(ar_list)
    ai_arr = np.array(ai_list)
    t_array = np.array(t_list)

    xmesh, ymesh = quad_mesh(x=z, y=t_array)
    plt.figure()
    plt.pcolormesh(xmesh, ymesh, ar_arr, cmap='YlGnBu')
    plt.axis(pad_limits(xmesh, ymesh))
    plt.colorbar()
    plt.xlabel('Ar')
    plt.ylabel('t')
    plt.title('title')
    plt.savefig('1d_nonlinear_Ar.png', dpi=600)

    plt.clf()
    plt.figure()
    plt.pcolormesh(xmesh, ymesh, ai_arr, cmap='YlGnBu')
    plt.axis(pad_limits(xmesh, ymesh))
    plt.colorbar()
    plt.xlabel('Ai')
    plt.ylabel('t')
    plt.title('title')
    plt.savefig('1d_nonlinear_Ai.png', dpi=600)
