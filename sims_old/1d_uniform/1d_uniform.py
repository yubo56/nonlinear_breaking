"""
Simplest 1D fluid equation in uniform medium

rho_1/rho = 0.001 seems to look linear, rho_1/rho = 0.1 produces steepening!
u << c, rho_1 << rho = linear
"""

import logging
import numpy as np
import matplotlib.pyplot as plt

from dedalus import public as de
from dedalus.extras.plot_tools import quad_mesh, pad_limits

logger = logging.getLogger(__name__)
XMAX = 10
N_X = 2048
DX = XMAX / N_X
DT = 1e-5
C = 5 # speed
T_F = 0.5 * XMAX / C

SIGMA = 0.3 # for the gaussian wave packet
WIDTH_GAUSS = 3 # how many sigma to go on either side of center of wave packet
RHO_0 = 10 # background rho
RAT = 0.5 # ratio u_1/C, rho_1/RHO_0

# Bases and domain
x_basis = de.Fourier('x', N_X, interval=(0, XMAX), dealias=3/2)
domain = de.Domain([x_basis], np.float64)

# Problem
problem = de.IVP(domain, variables=['u', 'r'])
problem.parameters['c'] = C
problem.add_equation("dt(r) = -r * dx(u) - u * dx(r)")
problem.add_equation("dt(u) = -c * c/r * dx(r) - u * dx(u)")

# Build solver
solver = problem.build_solver(de.timesteppers.SBDF2)
solver.stop_sim_time = T_F
solver.stop_wall_time = 100000 # should never get hit
solver.stop_iteration = 100000 # should never get hit

# Initial conditions
x = domain.grid(0)
u = solver.state['u']
r = solver.state['r']

u['g'] = [0] * len(x)
r['g'] = [10] * len(x)
for i in range(int(2 * WIDTH_GAUSS * SIGMA / DX)):
    center = WIDTH_GAUSS * SIGMA / DX
    u['g'][i] += C * RAT * np.exp(-(center - i) ** 2 / (2 * (SIGMA / DX) ** 2))
    r['g'][i] += RHO_0 * RAT * \
        np.exp(-(center - i) ** 2 / (2 * (SIGMA / DX) ** 2))

# Store data for final plot
u.set_scales(1, keep_data=True)
u_list = [np.copy(u['g'])]
r_list = [np.copy(r['g'])]
t_list = [solver.sim_time]

# Main loop
while solver.ok:
    solver.step(DT)
    if solver.iteration % 20 == 0:
        u.set_scales(1, keep_data=True)
        r.set_scales(1, keep_data=True)
        u_list.append(np.copy(u['g']))
        r_list.append(np.copy(r['g']))
        t_list.append(solver.sim_time)
    if solver.iteration % 1000 == 0:
        logger.info(
            'Iteration: %i, Time: %.3f/%3f, dt: %.3e',
            solver.iteration,
            solver.sim_time,
            T_F,
            DT
        )

# Create space-time plot
u_array = np.array(u_list)
r_array = np.array(r_list)
t_array = np.array(t_list)
xmesh, ymesh = quad_mesh(x=x, y=t_array)
plt.figure()
plt.pcolormesh(xmesh, ymesh, r_array, cmap='YlGnBu')
plt.axis(pad_limits(xmesh, ymesh))
plt.colorbar()
plt.xlabel('rho')
plt.ylabel('t')
plt.title('1D uniform wave equation, c=%.3f' % problem.parameters['c'])
plt.savefig('1d_uniform_rho.png', dpi=1200)
