'''
driving BC on left, rad BC on right, 1D wave eq. Test problem
'''

import logging
import numpy as np
import matplotlib.pyplot as plt

from dedalus import public as de
from dedalus.extras.plot_tools import quad_mesh, pad_limits

logger = logging.getLogger(__name__)

def plot_for_A(A):
    C = 5
    OMEGA = 2 * np.pi / 0.1

    SCALE = 32
    XMAX = 1
    N_X = 16
    T_F = (XMAX / C) * 6
    NUMSTEPS = 5e2
    DT = T_F / NUMSTEPS

    # Bases and domain
    x_basis = de.Chebyshev('x', N_X, interval=(0, XMAX))
    domain = de.Domain([x_basis], np.float64)

    # Problem
    sponge = domain.new_field()
    x = domain.grid(0)
    # sponge['g'] = np.zeros(np.shape(x))
    sponge['g'] = OMEGA * np.maximum(
        1 - (x - XMAX)**2 / (0.7 * XMAX - XMAX)**2,
        np.zeros(np.shape(x)))

    problem = de.IVP(domain, variables=['y', 'y_x', 'y_t'])
    problem.parameters['c'] = C
    problem.parameters['A'] = A
    problem.parameters['omega'] = OMEGA
    problem.parameters['sponge'] = sponge

    problem.add_equation('dt(y_t) - c**2 * dx(y_x) + sponge * y_t' +
                         '= 0')
    problem.add_equation('dx(y) - y_x = 0')
    problem.add_equation('dt(y) - y_t = 0')
    problem.add_bc('left(y) = A * cos(omega * t)')
    problem.add_bc('right(y_x + y_t / c) = 0')

    # Build solver
    solver = problem.build_solver(de.timesteppers.RK443)
    solver.stop_sim_time = T_F
    solver.stop_wall_time = 100000 # should never get hit
    solver.stop_iteration = 100000 # should never get hit

    # Initial conditions
    y = solver.state['y']
    y_x = solver.state['y_x']
    y_t = solver.state['y_t']
    gshape = domain.dist.grid_layout.global_shape(scales=1)

    y['g'] = np.zeros(gshape)
    y_x['g'] = np.zeros(gshape)
    y_t['g'] = np.zeros(gshape)

    # Store data for final plot
    t_list = [solver.sim_time]

    y.set_scales(SCALE, keep_data=True)
    y_x.set_scales(SCALE, keep_data=True)
    y_t.set_scales(SCALE, keep_data=True)
    y_list = [np.copy(y['g'])]
    yx_list = [np.copy(y_x['g'])]
    yt_list = [np.copy(y_t['g'])]

    # Main loop
    while solver.ok:
        solver.step(DT)
        if solver.iteration % (NUMSTEPS // 500) == 0:
            y.set_scales(SCALE, keep_data=True)
            y_x.set_scales(SCALE, keep_data=True)
            y_t.set_scales(SCALE, keep_data=True)
            y_list.append(np.copy(y['g']))
            yx_list.append(np.copy(y_x['g']))
            yt_list.append(np.copy(y_t['g']))
            t_list.append(solver.sim_time)
        if solver.iteration % (NUMSTEPS // 10) == 0:
            logger.info(
                'Iteration: %i, Time: %.3f/%3f, dt: %.3e',
                solver.iteration,
                solver.sim_time,
                T_F,
                DT
            )

    # Create space-time plot
    _x = np.array(domain.grid(0, scales=SCALE))
    trunc_len = int(len(_x) * 0.65)
    # trunc_len = len(_x)
    trunc_time = int((XMAX / C) * 2 / DT)
    x = _x[ : trunc_len]

    t_array = np.array(t_list)[trunc_time: ]
    y_array = np.array(y_list)[trunc_time: ]
    y_sub_array = y_array - np.array([
        A * np.cos(OMEGA * (_x / C - t)) for t in t_array])
    yx_array = np.array(yx_list)
    yt_array = np.array(yt_list)

    xmesh, tmesh = quad_mesh(x=x, y=t_array)
    fig = plt.figure()

    PLOT_CFG = True
    if PLOT_CFG:
        axes1 = fig.add_subplot(1, 2, 1, title='y')
        p1 = axes1.pcolormesh(xmesh, tmesh, y_array[ :, :trunc_len],
                              cmap='YlGnBu')
        axes1.axis(pad_limits(xmesh, tmesh))
        fig.colorbar(p1, ax=axes1)

        axes2 = fig.add_subplot(1, 2, 2, title='y - y_lin')
        p2 = axes2.pcolormesh(xmesh, tmesh, y_sub_array[ :, :trunc_len],
                              cmap='YlGnBu')
        axes2.axis(pad_limits(xmesh, tmesh))
        fig.colorbar(p2, ax=axes2)

    else:
        print(x[0:3])
        plt.plot(t_array, y_array[:, 0], 'g-')
        plt.plot(t_array, y_sub_array[:, 0], 'r-')

    fig.suptitle('Wave Equation, rad BC, A=%.3f' % A)
    fig.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.savefig('1d_rad_%.2f.png' % A, dpi=600)
    plt.clf()

if __name__ == '__main__':
    plot_for_A(0.1)
