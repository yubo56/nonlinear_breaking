"""
Testing 2d incompressible flow, Boussinesq (constant rho)
"""

import numpy as np
from mpi4py import MPI
import time

from dedalus import public as de
from dedalus.extras.plot_tools import quad_mesh, pad_limits

import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)


# Parameters
Lx, Lz = (4., 1.)
Prandtl = 1.
nu = 0.001

# Create bases and domain
x_basis = de.Fourier('x', 256, interval=(0, Lx), dealias=3/2)
z_basis = de.Chebyshev('z', 64, interval=(-Lz/2, Lz/2), dealias=3/2)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

# 2D Boussinesq hydrodynamics
problem = de.IVP(domain, variables=['P','u','w','uz','wz'])
problem.meta['P','u','w']['z']['dirichlet'] = True
problem.parameters['nu'] = nu
problem.add_equation("dx(u) + wz = 0")
problem.add_equation("dt(u) - nu*(dx(dx(u)) + dz(uz)) + dx(P) = -(u*dx(u) + w*uz)")
problem.add_equation("dt(w) - nu*(dx(dx(w)) + dz(wz)) + dz(P) = -(u*dx(w) + w*wz)")
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("wz - dz(w) = 0")
problem.add_bc("left(u) = 0")
problem.add_bc("left(w) = 0")
problem.add_bc("right(u) = 0")
problem.add_bc("right(w) = 0", condition="(nx != 0)")
problem.add_bc("right(P) = 0", condition="(nx == 0)")

# Build solver
solver = problem.build_solver(de.timesteppers.RK222)
logger.info('Solver built')

# Initial conditions
x = domain.grid(0)
z = domain.grid(1)
w = solver.state['w']
P = solver.state['P']
wz = solver.state['wz']

# Gaussian background
for i in range(len(w['g'])):
    # w['g'][i] += 0.001 *\
    #     np.ones(np.shape(w['g'][i])) * np.exp(-(z[0, :])**2 / (Lz ** 2 / 128)) * \
    #     np.cos(z[0, :] * 5) * \
    #     np.sin(x[i, 0] * 30)
    P['g'][i] += z[0, :]
w.differentiate('z', out=wz)

# Initial timestep
dt = 0.125

# Integration parameters
solver.stop_sim_time = 25
solver.stop_wall_time = 30 * 60.
solver.stop_iteration = 10

# Create abs(u) plot
xmesh, zmesh = quad_mesh(x=x[:,0], y=z[0])
plt.figure()
P.set_scales(1, keep_data=True)
plt.pcolormesh(xmesh, zmesh, np.transpose(P['g']), cmap='YlGnBu')
plt.axis(pad_limits(xmesh, zmesh))
plt.colorbar()
plt.xlabel('x')
plt.ylabel('z')
plt.title('Testing')
plt.savefig('test1.png')
plt.clf()

start_time = time.time()
while solver.ok:
    dt = solver.step(0.2)
    if (solver.iteration-1) % 10 == 0:
        logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))

plt.figure()
P.set_scales(1, keep_data=True)
plt.pcolormesh(xmesh, zmesh, np.transpose(P['g']), cmap='YlGnBu')
plt.axis(pad_limits(xmesh, zmesh))
plt.colorbar()
plt.xlabel('x')
plt.ylabel('z')
plt.title('Testing')
plt.savefig('test2.png')
