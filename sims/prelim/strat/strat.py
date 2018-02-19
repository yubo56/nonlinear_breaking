'''
Incompressible fluid equations w/
'''

import logging
import numpy as np
import matplotlib.pyplot as plt

from dedalus import public as de
from dedalus.extras.plot_tools import quad_mesh, pad_limits

logger = logging.getLogger(__name__)
XMAX = 10
ZMAX = 10
N_X = 64
N_Z = 64
DX = XMAX / N_X
DZ = ZMAX / N_Z
T_F = 20
DT = 1e-2

# Bases and domain
x_basis = de.Fourier('x', N_X, interval=(0, XMAX), dealias=3/2)
z_basis = de.Chebyshev('z', N_Z, interval=(0, ZMAX), dealias=3/2)
domain = de.Domain([x_basis, z_basis], np.float64)

problem = de.IVP(domain, variables=['P', 'ux', 'uz'])
problem.meta['P']['z']['dirichlet'] = True
problem.parameters['rho0'] = 1
problem.add_equation("dx(ux) + dz(uz) = 0")
problem.add_equation("dt(ux) + dx(P)/rho0 = 0")
problem.add_equation("dt(uz) + dz(P)/rho0 = 0")
problem.add_bc("right(dt(P) - dz(P)) = 0")
problem.add_bc("left(P) = sin(6.28 * (x - t) / 10)")

# Build solver
solver = problem.build_solver(de.timesteppers.RK222)
solver.stop_sim_time = T_F
solver.stop_wall_time = np.inf
solver.stop_iteration = 100

# Initial conditions
x = domain.grid(0)
z = domain.grid(1)
ux = solver.state['ux']
uz = solver.state['uz']
P = solver.state['P']

# need to reverse for some reason?
gshape = domain.dist.grid_layout.global_shape(scales=1)
P['g'] = np.zeros(gshape)
ux['g'] = np.zeros(gshape)
uz['g'] = np.zeros(gshape)

# Main loop
for stop_iter in [50, 100]:
    solver.stop_iteration = stop_iter
    while solver.ok:
        solver.step(DT)

    ux.set_scales(1, keep_data=True)
    uz.set_scales(1, keep_data=True)
    P.set_scales(1, keep_data=True)

    # plot vars
    xmesh, zmesh = quad_mesh(x=x[:,0], y=z[0])
    for var, name in [(ux['g'], 'ux'), (uz['g'], 'uz'), (P['g'], 'P')]:
        plt.figure()
        plt.pcolormesh(xmesh, zmesh, np.transpose(var), cmap='YlGnBu')
        plt.axis(pad_limits(xmesh, zmesh))
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('z')
        plt.title('%s, (t = %d)' % (name, stop_iter))
        plt.savefig('no_g_%s_t%d.png' % (name, stop_iter))
        plt.clf()
