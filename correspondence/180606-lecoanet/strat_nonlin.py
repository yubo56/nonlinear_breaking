from collections import defaultdict
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

from dedalus import public as de
from dedalus.extras.flow_tools import CFL
from dedalus.extras.plot_tools import quad_mesh, pad_limits

def get_omega(g, h, kx, kz):
    return np.sqrt((g / h) * kx**2 / (kx**2 + kz**2 + 0.25 / h**2))

def get_vph(g, h, kx, kz):
    norm = abs(get_omega(g, h, kx, kz)) / (kx**2 + kz**2)
    return norm * kz, norm * kz

H = 1
XMAX = H
ZMAX = 5 * H
KX = 2 * np.pi / H
KZ = -(np.pi / 2) * np.pi / H
G = 1

OMEGA = get_omega(G, H, KX, KZ)
VPH_X, VPH_Z = get_vph(G, H, KX, KZ)

num_timesteps = 2e3
DAMP_START = ZMAX * 0.7 # start damping zone
T_F = -ZMAX / VPH_Z # VPH_Z < 0
DT = T_F / num_timesteps
N_X = 16
N_Z = 64
RHO0 = 1
A = 0.05
NUM_SNAPSHOTS = 1e3

def get_analytical_sponge(name, z_pts, t):
    uz_anal = A * np.exp(z_pts / (2 * H)) *\
        np.cos(KZ * z_pts - OMEGA * t)
    rho0 = RHO0 * np.exp(-z_pts / H)
    analyticals = {
        'uz': uz_anal,
        'ux': -KZ / KX * uz_anal,
        'rho': -rho0 * A / (H * OMEGA) *\
            np.exp(z_pts / (2 * H)) *\
            np.sin(KZ * z_pts - OMEGA * t),
        'P': -rho0 * OMEGA / KX**2 * KZ *\
            uz_anal,
    }
    return analyticals[name]

if __name__ == '__main__':
    logger = logging.getLogger(__name__)

    x_basis = de.Fourier('x', N_X, interval=(0, XMAX), dealias=3/2)
    z_basis = de.Chebyshev('z', N_Z, interval=(0, ZMAX), dealias=3/2)
    domain = de.Domain([x_basis, z_basis], np.float64)
    x = domain.grid(0)
    z = domain.grid(1)

    problem = de.IVP(domain, variables=['P', 'rho', 'ux', 'uz'])
    problem.meta['uz']['z']['dirichlet'] = True
    problem.parameters['L'] = XMAX
    problem.parameters['g'] = G
    problem.parameters['H'] = H
    problem.parameters['A'] = A
    problem.parameters['KX'] = KX
    problem.parameters['KZ'] = KZ
    problem.parameters['omega'] = get_omega(G, H, KX, KZ)

    # rho0 stratification
    rho0 = domain.new_field()
    rho0.meta['x']['constant'] = True
    rho0['g'] = RHO0 * np.exp(-z / H)
    problem.parameters['rho0'] = rho0

    sponge_strength = 1
    z = domain.grid(1)

    # sponge field
    sponge = domain.new_field()
    sponge.meta['x']['constant'] = True
    sponge['g'] = sponge_strength * (
        np.maximum(z - DAMP_START, 0) ** 2 / (ZMAX - DAMP_START) ** 2)

    problem.parameters['sponge'] = sponge
    problem.add_equation('dx(ux) + dz(uz) = 0')
    problem.add_equation('dt(rho) - rho0 * uz / H + sponge * rho'
                         + '= -ux * dx(rho) - uz * dz(rho)')
    problem.add_equation('dt(ux) + dx(P) / rho0 + sponge * ux'
                         + '= -ux * dx(ux) - uz * dz(ux)')
    problem.add_equation('dt(uz) + dz(P) / rho0 + rho * g / rho0 + sponge * uz'
                         + '= -ux * dx(uz) - uz * dz(uz)')

    problem.add_bc('left(P) = 0', condition='nx == 0')
    problem.add_bc('right(uz) = 0', condition='nx != 0')
    problem.add_bc(
        'left(dz(uz)) = -KZ * A * sin(KX * x - omega * t + 1 / (2 * H))',
        condition='nx != 0')
    problem.add_bc('left(uz) = 0', condition='nx == 0')

    # Build solver
    solver = problem.build_solver(de.timesteppers.RK443)
    solver.stop_sim_time = T_F
    solver.stop_wall_time = np.inf
    solver.stop_iteration = np.inf

    # Initial conditions
    ux = solver.state['ux']
    uz = solver.state['uz']
    P = solver.state['P']
    rho = solver.state['rho']
    gshape = domain.dist.grid_layout.global_shape(scales=1)

    P['g'] = np.zeros(gshape)
    ux['g'] = np.zeros(gshape)
    uz['g'] = np.zeros(gshape)
    rho['g'] = np.zeros(gshape)

    dyn_vars = ['uz', 'ux', 'rho', 'P']
    state_vars = defaultdict(list)
    sim_times = []
    rho0 = RHO0 * np.exp(-z / H)

    # Main loop
    try:
        logger.info('Starting sim...')
        while solver.ok:
            if solver.iteration % int((T_F / DT) / NUM_SNAPSHOTS) == 0:
                logger.info('Reached time %f out of %f' %
                      (solver.sim_time, solver.stop_sim_time))
                sim_times.append(solver.sim_time)
                for varname in dyn_vars:
                    values = solver.state[varname]
                    values.set_scales(1, keep_data=True)
                    state_vars[varname].append(np.copy(values['g']))

            solver.step(DT)
    except:
        pass

    # plot results
    matplotlib.rcParams.update({'font.size': 6})
    try:
        os.makedirs('plots_nonlin')
    except FileExistsError:
        pass

    for varname in dyn_vars:
        # truncate for overflow
        state_vars[varname] = np.array(state_vars[varname])[:-1]
    state_vars['E'] = np.sum(
        ((rho0 + state_vars['rho']) *
         (state_vars['ux']**2 + state_vars['uz']**2)) / 2,
        axis=1)
    state_vars['F_z'] = np.sum(
        state_vars['uz'] * ((rho0 + state_vars['rho']) *
                            (state_vars['ux']**2 + state_vars['uz']**2) +
                            state_vars['P']),
        axis=1)

    mesh_vars = ['uz', 'ux']
    slice_vars = ['uz', 'ux', 'P', 'rho']
    z_vars = ['F_z', 'E']
    n_cols = 3
    n_rows = 3
    xmesh, zmesh = quad_mesh(x=x[:, 0], y=z[0])
    for t_idx, sim_time in list(enumerate(sim_times[:-1])):
        fig = plt.figure(dpi=200)

        idx = 1
        for var in mesh_vars:
            axes = fig.add_subplot(n_rows, n_cols, idx, title=var)

            var_dat = state_vars[var]
            p = axes.pcolormesh(xmesh,
                                zmesh,
                                var_dat[t_idx].T)
                                # vmin=var_dat.min(), vmax=var_dat.max())
            axes.axis(pad_limits(xmesh, zmesh))
            cb = fig.colorbar(p, ax=axes)
            cb.ax.set_yticklabels(cb.ax.get_yticklabels(), rotation=30)
            plt.xticks(rotation=30)
            plt.yticks(rotation=30)
            idx += 1
        for var in z_vars + slice_vars:
            axes = fig.add_subplot(n_rows, n_cols, idx, title=var)
            z_pts = (zmesh[1:, 0] + zmesh[:-1, 0]) / 2

            if var in slice_vars:
                var_dat = state_vars[var][:, 0, :]
                # p = axes.plot(get_analytical_sponge(var, z_pts, sim_time),
                #               z_pts)
            else:
                var_dat = state_vars[var]

            p = axes.plot(var_dat[t_idx], z_pts)
            plt.xticks(rotation=30)
            plt.yticks(rotation=30)
            # xlims = [var_dat.min(), var_dat.max()]
            xlims = [var_dat[t_idx].min(), var_dat[t_idx].max()]
            axes.set_xlim(*xlims)
            p = axes.plot(xlims, [DAMP_START] * len(xlims), 'r--')
            idx += 1

        fig.suptitle('t=%.2f, kx=-2pi/H, kz=2pi/H, omega=%.2f' %
                     (sim_time, OMEGA))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        savefig = 'plots_nonlin/t_%d.png' % (t_idx)
        plt.savefig(savefig)
        logger.info('Saved %s' % savefig)
        plt.close()
