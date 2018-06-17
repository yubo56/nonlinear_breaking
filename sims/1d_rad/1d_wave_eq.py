'''
1d radiative BCs, toy problem to understand volumetric forcing
sigma = (1,2,3), y_max = (0.839, 2.319, 3.729)
'''

import logging
import numpy as np
import matplotlib.pyplot as plt

from dedalus import public as de
from dedalus.extras.plot_tools import quad_mesh, pad_limits

logger = logging.getLogger(__name__)

def plot_for_sigma(SIGMA):
    SCALE = 4
    XMAX = 30
    N_X = 128
    T_F = 100
    NUMSTEPS = 5e2
    DT = T_F / NUMSTEPS

    # Bases and domain
    x_basis = de.Chebyshev('x', N_X, interval=(0, XMAX), dealias=3/2)
    domain = de.Domain([x_basis], np.float64)

    # Problem
    x = domain.grid(0)

    problem = de.IVP(domain, variables=['y', 'y_x', 'y_t'])

    problem.parameters['SIGMA'] = SIGMA
    problem.add_equation('dt(y_t) - dx(y_x)' +
                         '= cos(x - t) * exp(-(x - 10)**2 / (2 * SIGMA**2)) / SIGMA')
    problem.add_equation('dx(y) - y_x = 0')
    problem.add_equation('dt(y) - y_t = 0')
    problem.add_bc('right(y_x + y_t) = 0')
    problem.add_bc('left(y_x - y_t) = 0')

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
        if solver.iteration % (NUMSTEPS // 2) == 0:
            logger.info(
                'Iteration: %i, Time: %.3f/%3f, dt: %.3e',
                solver.iteration,
                solver.sim_time,
                T_F,
                DT
            )

    # Create space-time plot
    _x = np.array(domain.grid(0, scales=SCALE))

    t_array = np.array(t_list)
    y_array = np.array(y_list)
    yx_array = np.array(yx_list)
    yt_array = np.array(yt_list)
    print('Max:', y_array.max())

    xmesh, tmesh = quad_mesh(x=_x, y=t_array)
    fig = plt.figure()

    PLOT_CFG = True
    if PLOT_CFG:
        axes1 = fig.add_subplot(1, 1, 1, title='y')
        p1 = axes1.pcolormesh(xmesh, tmesh, y_array,
                              cmap='YlGnBu')
        axes1.axis(pad_limits(xmesh, tmesh))
        fig.colorbar(p1, ax=axes1)

    else:
        plt.plot(t_array, y_array[:, 0], 'g-')

    fig.suptitle('Wave Equation, rad BC')
    fig.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.savefig('1d_rad_%d.png' % SIGMA, dpi=600)
    plt.clf()

if __name__ == '__main__':
    plot_for_sigma(1)
    plot_for_sigma(2)
    plot_for_sigma(3)
    plot_for_sigma(4)
    plot_for_sigma(5)
