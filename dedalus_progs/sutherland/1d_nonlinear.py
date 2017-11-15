"""
Eq. 26 from Sutherland et al 2011
"""

import logging
import numpy as np
import matplotlib.pyplot as plt

from dedalus import public as de
from dedalus.extras.plot_tools import quad_mesh, pad_limits

logger = logging.getLogger(__name__)
WIDTH = 2 # go out to 2 sigma on either side
K = 1
N = 1
ZMIN = -20 / K
ZMAX = 40 / K
N_X = 2048
DX = (ZMAX - ZMIN) / N_X
DT = 1e-3 / N

m = -0.4 * K
w = 0.93 * N
C_G = 0.32 * N / K
w_mm = -0.47 * N / K ** 2
w_mmm = -1.91 * N / K ** 3
H = 10 / K
sigm = 10 / K
A = 0.05 / K

T_F = 5 / N * 0.01

# Bases and domain
x_basis = de.Chebyshev('z', N_X, interval=(ZMIN, ZMAX))
domain = de.Domain([x_basis], np.float64)

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
problem.meta[:]['z']['dirichlet'] = True
problem.add_equation('dt(Ar) = -C_G * dz(Ar) - w_mm * dz(dz(Ai)) / 2 + \
    w_mmm * dz(dz(dz(Ar))) / 6 - K * U0 * (Ar**2 + Ai**2) * e**(z/H) * Ar + \
    w**2 / (2 * N**2 * K * H) * (U0 * e**(z/H) * \
    (2 * Ar * dz(Ar) + 2 * Ai * dz(Ai))) * (3 * m * H * Ar + Ai)')
problem.add_equation('dt(Ai) = -C_G * dz(Ai) + w_mm * dz(dz(Ar)) / 2 + \
    w_mmm * dz(dz(dz(Ai))) / 6 + K * U0 * (Ar**2 + Ai**2) * e**(z/H) * Ai + \
    w**2 / (2 * N**2 * K * H) * (U0 * e**(z/H) * \
    (2 * Ar * dz(Ar) + 2 * Ai * dz(Ai))) * (3 * m * H * Ai - Ar)')

# Build solver
solver = problem.build_solver(de.timesteppers.SBDF2)
solver.stop_sim_time = T_F
solver.stop_wall_time = 100000 # should never get hit
solver.stop_iteration = 100000 # should never get hit

# Initial conditions
z = domain.grid(0)
zero_idx = -ZMIN / DX
if zero_idx < WIDTH * sigm / DX:
    raise ValueError('Domain cannot fit Gaussian of width %d centered at %d'
            % (WIDTH * sigm / DX, zero_idx))
Ar = solver.state['Ar']
Ai = solver.state['Ai']

Ar['g'] = np.zeros(np.shape(z))
Ai['g'] = np.zeros(np.shape(z))
for i in range(int(zero_idx - WIDTH * sigm / DX),
        int(zero_idx + WIDTH * sigm / DX)):
    Ar['g'][i] += A * np.exp(-(zero_idx - i)**2 / (2 * (sigm / DX)**2))

# Store data for final plot
Ar.set_scales(1, keep_data=True)
Ai.set_scales(1, keep_data=True)
ar_list = [np.copy(Ar['g'])]
ai_list = [np.copy(Ai['g'])]
t_list = [solver.sim_time]

# Main loop
while solver.ok:
    solver.step(DT)
    if solver.iteration % 20 == 0:
        Ar.set_scales(1, keep_data=True)
        Ai.set_scales(1, keep_data=True)
        ar_list.append(np.copy(Ar['g']))
        ai_list.append(np.copy(Ai['g']))
        t_list.append(solver.sim_time)
    if solver.iteration % 100 == 0:
        logger.info(
            'Iteration: %i, Time: %.3f/%3f, dt: %.3e',
            solver.iteration,
            solver.sim_time,
            T_F,
            DT
        )

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
plt.savefig('1d_nonlinear.png', dpi=600)
