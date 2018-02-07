'''
Linear Incompressible 2D. EOM are:
    - Du/Dt = -Div P/rho^(0) - g
    - Div(u) = 0
    - D(rho)/Dt = 0
Things start propto e^{-z^2 / (2*s^2)} cos(kx + mz)e^{-z/2H}
    - k * s = 10
    - k * H = 10
    - k * A (displacement) = 0.05
    - m = -0.4k
- Things should be periodic-ish in x, Fourier basis
    - less so in z, chebyshev (like example)
'''

import logging
import numpy as np
import matplotlib.pyplot as plt

from dedalus import public as de
from dedalus.extras.plot_tools import quad_mesh, pad_limits

logger = logging.getLogger(__name__)
XMAX = 10
ZMAX = 100
N_X = 8
N_Z = 8
DX = XMAX / N_X
DZ = ZMAX / N_Z
T_F = 1
DT = 1e-4

rho0 = 1 # kg/m^3
P0 = 100 # Pa
q0 = 1e5 # K
k = 2
S = 10 / k
H = 10 / k
A = 0.05 / k
m = -0.4 * k
g = 10
nu = 2e-5

# Bases and domain
x_basis = de.Fourier('x', N_X, interval=(0, XMAX), dealias=3/2)
z_basis = de.Chebyshev('z', N_Z, interval=(0, ZMAX), dealias=3/2)
domain = de.Domain([x_basis, z_basis], np.float64)

# Problem
problem = de.IVP(domain, variables=['u', 'uz', 'w', 'wz', 'P'])
problem.parameters['g'] = g
# problem.parameters['rho0'] = rho0
# problem.parameters['H'] = H
problem.add_equation('dx(u) + wz = 0')
problem.add_equation('dt(u) + dx(P) = -(u * dx(u) + w * uz)')
problem.add_equation('dt(w) + dz(P) = -(u * dx(w) + w * wz) + g')
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("wz - dz(w) = 0")
problem.add_bc('right(w) = 0')
problem.add_bc('right(u) = 0')
problem.add_bc('right(P) = 0')
# need one BC per occurrence of dz (chebyshev)

# Build solver
solver = problem.build_solver(de.timesteppers.RK222)
solver.stop_sim_time = T_F
solver.stop_wall_time = 100000 # should never get hit
solver.stop_iteration = 10 # should never get hit

# Initial conditions
x = domain.grid(0)
z = domain.grid(1)
u = solver.state['u']
uz = solver.state['uz']
w = solver.state['w']
wz = solver.state['wz']
P = solver.state['P']

# need to reverse for some reason?
gshape = domain.dist.grid_layout.global_shape(scales=1)
print('=====2d_linear.py=====', 'gshape', gshape);
P['g'] = P0 * np.ones(gshape)
u['g'] = np.ones(gshape)
w['g'] = np.ones(gshape)

for i in range(gshape[0]):
    for j in range(max(gshape[1], 4 * S / DZ)):
        x_coord = i * DX
        z_coord = j * DZ - 2 * S
        u['g'][i][j] += A * m * np.exp(-z_coord**2 / (2 * S**2))\
            * np.cos(k * x_coord + m * z_coord) * np.exp(-z_coord / (2 * H))
        w['g'][i][j] += A * k * np.exp(-z_coord**2 / (2 * S**2))\
            * np.cos(k * x_coord + m * z_coord) * np.exp(-z_coord / (2 * H))

# u.differentiate('z', out=uz)
# w.differentiate('z', out=wz)

# Create abs(u) plot
xmesh, zmesh = quad_mesh(x=x[:,0], y=z[0])
absu = np.transpose(np.sqrt(u['g']**2 + w['g']**2))
print(np.shape(absu))
plt.figure()
plt.pcolormesh(xmesh, zmesh, absu, cmap='YlGnBu')
plt.axis(pad_limits(xmesh, zmesh))
plt.colorbar()
plt.xlabel('x')
plt.ylabel('z')
plt.title('Testing')
plt.show()

# Store data for final plot
u.set_scales(1, keep_data=True)
w.set_scales(1, keep_data=True)
u_list = [np.copy(u['g'])]
w_list = [np.copy(w['g'])]
t_list = [solver.sim_time]

# Main loop
for i in range(1):
    solver.step(DT)
    u.set_scales(1, keep_data=True)
    w.set_scales(1, keep_data=True)
    u_list.append(np.copy(u['g']))
    w_list.append(np.copy(w['g']))
    t_list.append(solver.sim_time)
    if solver.iteration % 1000 == 0:
        logger.info(
            'Iteration: %i, Time: %.3f/%3f, dt: %.3e',
            solver.iteration,
            solver.sim_time,
            T_F,
            DT
        )

# Create abs(u) plot
xmesh, zmesh = quad_mesh(x=x[:,0], y=z[0])
absu = np.transpose(np.sqrt(u_list[1]**2 + w_list[1]**2))
plt.figure()
plt.pcolormesh(xmesh, zmesh, absu, cmap='YlGnBu')
plt.axis(pad_limits(xmesh, zmesh))
plt.colorbar()
plt.xlabel('x')
plt.ylabel('z')
plt.title('Testing')
plt.show()
