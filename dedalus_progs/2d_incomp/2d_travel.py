'''
Incompressible fluid equations w/o delta rho, velocity perturbation just travels
'''

import logging
import numpy as np
import matplotlib.pyplot as plt

from dedalus import public as de
from dedalus.extras.plot_tools import quad_mesh, pad_limits

logger = logging.getLogger(__name__)
XMAX = 10
ZMAX = 100
N_X = 64
N_Z = 256
DX = XMAX / N_X
DZ = ZMAX / N_Z
T_F = 20
DT = 1e-2

rho0 = 1 # kg/m^3
P0 = 100 # Pa
q0 = 1e5 # K
k = 2
S = 10 / k
H = 10 / k
A = 0.05 / k
m = -0.4 * k
g = P0 / (H * rho0)
nu = 2e-3

# Bases and domain
x_basis = de.Fourier('x', N_X, interval=(0, XMAX), dealias=3/2)
z_basis = de.Chebyshev('z', N_Z, interval=(0, ZMAX), dealias=3/2)
domain = de.Domain([x_basis, z_basis], np.float64)

problem = de.IVP(domain, variables=['P', 'u', 'w', 'uz', 'wz'])
problem.meta['P','u','w']['z']['dirichlet'] = True
problem.parameters['nu'] = nu
problem.parameters['H'] = H
problem.parameters['g'] = g
problem.parameters['rho0'] = rho0
problem.add_equation("dx(u) + wz = 0")
# problem.add_equation('dt(rho) = -u * dx(rho) - w * dz(rho)')
problem.add_equation(
    "dt(u) - nu*(dx(dx(u)) + dz(uz)) + dx(P)/rho0 = -(u*dx(u) + w*uz)")
problem.add_equation(
    "dt(w) - nu*(dx(dx(w)) + dz(wz)) + dz(P)/rho0 = -(u*dx(w) + w*wz) - g")
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("wz - dz(w) = 0")
problem.add_bc("left(u) = 0")
problem.add_bc("left(w) = 0", condition="(nx != 0)")
problem.add_bc("left(P) = rho0 * g * 100", condition="(nx == 0)")
problem.add_bc("right(u) = 0")
problem.add_bc("right(w) = 0", condition="(nx != 0)")
problem.add_bc("right(P) = 0", condition="(nx == 0)")

# Build solver
solver = problem.build_solver(de.timesteppers.RK222)
solver.stop_sim_time = T_F
solver.stop_wall_time = 100000 # should never get hit
solver.stop_iteration = 100000 # should never get hit

# Initial conditions
x = domain.grid(0)
z = domain.grid(1)
u = solver.state['u']
uz = solver.state['uz']
w = solver.state['w']
wz = solver.state['wz']
P = solver.state['P']
# rho = solver.state['rho']

# need to reverse for some reason?
gshape = domain.dist.grid_layout.global_shape(scales=1)
# rho['g'] = rho0 * np.ones(gshape)
u['g'] = np.zeros(gshape)
w['g'] = np.ones(gshape)

for i in range(gshape[0]):
    for j in range(gshape[1]):
        x_coord = i * DX
        z_coord = j * DZ - ZMAX / 2
        u['g'][i][j] += A * m * np.exp(-z_coord**2 / (2 * S**2))\
            * np.cos(k * x_coord + m * z_coord) * np.exp(-z_coord / (2 * H))
        w['g'][i][j] += A * k * np.exp(-z_coord**2 / (2 * S**2))\
            * np.cos(k * x_coord + m * z_coord) * np.exp(-z_coord / (2 * H))
for j in range(N_Z):
    P['g'][:, j] = np.ones(np.shape(P['g'][:, j])) * rho0 * g * (N_Z - j) * DZ

u.differentiate('z', out=uz)
w.differentiate('z', out=wz)

# Create abs(u) plot
xmesh, zmesh = quad_mesh(x=x[:,0], y=z[0])
u.set_scales(1, keep_data=True)
w.set_scales(1, keep_data=True)
P.set_scales(1, keep_data=True)
absu = np.transpose(np.sqrt(u['g']**2 + w['g']**2))
plt.figure()
plt.pcolormesh(xmesh, zmesh, absu, cmap='YlGnBu')
plt.axis(pad_limits(xmesh, zmesh))
plt.colorbar()
plt.xlabel('x')
plt.ylabel('z')
plt.title('Testing')
plt.savefig('travel0.png')
plt.clf()

# Store data for final plot
u.set_scales(1, keep_data=True)
w.set_scales(1, keep_data=True)
u_list = [np.copy(u['g'])]
w_list = [np.copy(w['g'])]
t_list = [solver.sim_time]

# Main loop
idx = 0
while solver.ok:
    solver.step(DT)
    u.set_scales(1, keep_data=True)
    w.set_scales(1, keep_data=True)
    P.set_scales(1, keep_data=True)
    u_list.append(np.copy(u['g']))
    w_list.append(np.copy(w['g']))
    t_list.append(solver.sim_time)
    if solver.iteration % 500 == 0:
        maxu = np.sqrt(u['g']**2 + w['g']**2).max()
        idx += 1
        logger.info(
            '''Iteration: %i, Time: %.3f/%3f, dt: %.3e
            \tSaving travel%s.png. Max velocity is %.3f''',
            solver.iteration,
            solver.sim_time,
            T_F,
            DT,
            idx,
            maxu
        )

        # Create abs(u) plot
        xmesh, zmesh = quad_mesh(x=x[:,0], y=z[0])
        absu = np.transpose(np.sqrt(u_list[-1]**2 + w_list[-1]**2))
        plt.figure()
        plt.pcolormesh(xmesh, zmesh, absu, cmap='YlGnBu')
        plt.axis(pad_limits(xmesh, zmesh))
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('z')
        plt.title('Testing')
        plt.savefig('travel%s.png' % idx)
        plt.clf()
