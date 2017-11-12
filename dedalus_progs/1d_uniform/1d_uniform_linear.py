"""
Simplest 1D fluid equation in uniform medium
NOTE: (XMAX/N_X) / (C * DT) should be >> 1 for stability, seems to start failing
around 10?
"""

import logging
import numpy as np
import matplotlib.pyplot as plt

from dedalus import public as de
from dedalus.extras.plot_tools import quad_mesh, pad_limits

logger = logging.getLogger(__name__)
XMAX = 10
N_X = 512
DX = XMAX / N_X
DT = 1e-3
C = 1
SIGMA = 0.3 # for the gaussian wave packet
WIDTH_GAUSS = 3 # how many sigma to go on either side of center of wave packet

# Bases and domain
x_basis = de.Fourier('x', N_X, interval=(0, XMAX), dealias=0)
domain = de.Domain([x_basis], np.float64)

# Problem
problem = de.IVP(domain, variables=['u', 'r'])
problem.parameters['c'] = C
problem.parameters['r0'] = 1
problem.add_equation("dt(r) + r0 * dx(u) = 0")
problem.add_equation("dt(u) = -c * c/r0 * dx(r)")

# Build solver
solver = problem.build_solver(de.timesteppers.SBDF2)
solver.stop_sim_time = XMAX / C
solver.stop_iteration = 60000 # should never get hit

# Initial conditions
x = domain.grid(0)
u = solver.state['u']
r = solver.state['r']

u['g'] = 0 * x
r['g'] = 0 * x
for i in range(int(2 * WIDTH_GAUSS * SIGMA / DX)):
    center = WIDTH_GAUSS * SIGMA / DX
    u['g'][i] = np.exp(-(center - i) ** 2 / (2 * (SIGMA / DX) ** 2))
    r['g'][i] = u['g'][i]

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
    if solver.iteration % 200 == 0:
        logger.info(
            'Iteration: %i, Time: %.3f, dt: %.3f',
            solver.iteration,
            solver.sim_time,
            DT
        )

# Create space-time plot
u_array = np.array(u_list)
r_array = np.array(r_list)
t_array = np.array(t_list)
xmesh, ymesh = quad_mesh(x=x, y=t_array)
print(np.shape(xmesh), np.shape(ymesh), np.shape(x), np.shape(t_array),
    np.shape(r_array))
plt.figure()
plt.pcolormesh(xmesh, ymesh, r_array, cmap='winter')
plt.axis(pad_limits(xmesh, ymesh))
plt.colorbar()
plt.xlabel('rho')
plt.ylabel('t')
plt.title('1D uniform wave equation, c=%.3f' % problem.parameters['c'])
plt.savefig('1d_uniform_linear_rho.png')
