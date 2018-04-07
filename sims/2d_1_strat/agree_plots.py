#!/usr/bin/env python
# plot agreement of simulation/analytical at various (x, t) slices over z.

import os
import h5py
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import strat_helper

A = 0.05
H = 1
XMAX = H
ZMAX = 5 * H
KX = 2 * np.pi / H
_KZ = (np.pi / 2) * np.pi / H
G = 10
RHO0 = 1
OMEGA = strat_helper.get_omega(G, H, KX, _KZ)

def agree_sponge(plot_idx, t_idx=-30):
    KZ = -_KZ
    dat = h5py.File('snapshots_sponge/snapshots_sponge_s1/snapshots_sponge_s1_p0.h5', mode='r')
    tmesh = np.array(dat['tasks']['uz'].dims[0][0])
    xmesh = np.array(dat['tasks']['uz'].dims[1][0])
    zmesh = np.array(dat['tasks']['uz'].dims[2][0])

    x_idx = len(xmesh) // 2

    uz_anal = A * np.exp(zmesh / (2 * H)) \
        * np.cos(KX * xmesh[x_idx] + KZ * zmesh - OMEGA * tmesh[t_idx])
    rho0 = RHO0 * np.exp(-zmesh / H)
    analyticals = {
        'uz': uz_anal,
        'ux': KZ / KX * uz_anal,
        'rho': -rho0 * A / (H * OMEGA)* np.exp(zmesh / (2 * H)) \
            * np.sin(KX * xmesh[x_idx] + KZ * zmesh - OMEGA * tmesh[t_idx]),
        'P': -rho0 * OMEGA / KX**2 * KZ * uz_anal,
    }

    fig = plt.figure(dpi=200)
    for idx, var in enumerate(['uz', 'ux', 'rho', 'P']):
        axes = fig.add_subplot(2, 2, idx + 1, title=var)
        vardat = dat['tasks'][var]
        sliced = vardat[t_idx, x_idx, :]
        axes.plot(zmesh, sliced, label='Numerical')

        axes.plot(zmesh,
                 analyticals[var],
                 label='Analytical'
                )
        axes.legend()
    fig.suptitle('Sponge (t=%.3f, x=%.3f) (damping zone z>=3.875)' %
                 (tmesh[t_idx], xmesh[x_idx]))
    plt.savefig('agree_plots/sponge_%d.png' % plot_idx)
    plt.clf()
    return plot_idx + 1

def agree_d0(plot_idx, t_idx=-10):
    KZ = _KZ
    dat = h5py.File('snapshots_d0/snapshots_d0_s1/snapshots_d0_s1_p0.h5', mode='r')
    tmesh = np.array(dat['tasks']['uz'].dims[0][0])
    xmesh = np.array(dat['tasks']['uz'].dims[1][0])
    zmesh = np.array(dat['tasks']['uz'].dims[2][0])

    x_idx = len(xmesh) // 2

    uz_anal = A * np.exp(zmesh / (2 * H)) \
        * np.cos(KX * xmesh[x_idx] - OMEGA * tmesh[t_idx]) \
        * np.sin(KZ * (ZMAX - zmesh)) / np.sin(KZ * ZMAX)
    rho0 = RHO0 * np.exp(-zmesh / H)
    analyticals = {
        'uz': uz_anal,
        'ux': KZ / KX * A * np.exp(zmesh / (2 * H)) \
            * np.sin(KX * xmesh[x_idx] - OMEGA * tmesh[t_idx]) \
            * np.cos(KZ * (ZMAX - zmesh)) / np.sin(KZ * ZMAX),
        'rho': -rho0 * A / (H * OMEGA) * np.exp(zmesh / (2 * H)) \
            * np.sin(KX * xmesh[x_idx] - OMEGA * tmesh[t_idx]) \
            * np.sin(KZ * (ZMAX - zmesh)) / np.sin(KZ * ZMAX),
        'P': -rho0 * OMEGA / KX**2 * KZ * A * np.exp(zmesh / (2 * H)) \
            * np.cos(KX * xmesh[x_idx] - OMEGA * tmesh[t_idx]) \
            * np.sin(KZ * (ZMAX - zmesh)) / np.sin(KZ * ZMAX),
    }


    fig = plt.figure(dpi=200)
    for idx, var in enumerate(['uz', 'ux', 'rho', 'P']):
        axes = fig.add_subplot(2, 2, idx + 1, title=var)
        vardat = dat['tasks'][var]
        sliced = vardat[t_idx, x_idx, :]
        axes.plot(zmesh, sliced, label='Numerical')

        axes.plot(zmesh,
                 analyticals[var],
                 label='Analytical'
                )
        axes.legend()
    fig.suptitle('Dirichlet (t=%.3f, x=%.3f)' %
                 (tmesh[t_idx], xmesh[x_idx]))
    plt.savefig('agree_plots/d0_%d.png' % plot_idx)
    plt.clf()
    return plot_idx + 1

if __name__ == '__main__':
    try:
        os.makedirs('agree_plots')
    except FileExistsError:
        pass

    matplotlib.rcParams.update({'font.size': 6})
    plot_idx = 0

    plot_idx = agree_sponge(plot_idx, -4)
    plot_idx = agree_sponge(plot_idx, -3)
    plot_idx = agree_sponge(plot_idx, -2)
    plot_idx = agree_sponge(plot_idx, -1)

    plot_idx = 0
    plot_idx = agree_d0(plot_idx, -4)
    plot_idx = agree_d0(plot_idx, -3)
    plot_idx = agree_d0(plot_idx, -2)
    plot_idx = agree_d0(plot_idx, -1)
