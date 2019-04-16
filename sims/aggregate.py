'''
using data outputted from 2d_3_final, 2d_4_fourier on refl/Ri
'''
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
import matplotlib.lines as mlines

DATA = [
    (0.70, ((0.1129, 0.0950, 0.1267), (0.1416, 0.1220, 0.1560), (0.0001, 0.0000, 0.0006), (0.4834, 0.4486, 0.5716))),
    (0.50, ((0.1737, 0.1541, 0.1919), (0.2478, 0.2159, 0.2763), (0.0040, 0.0017, 0.0063), (0.2965, 0.2481, 0.3464))),
    (0.30, ((0.2659, 0.2046, 0.3090), (0.4193, 0.4122, 0.4270), (0.0000, 0.0000, 0.0002), (0.4387, 0.3322, 0.5149))),
    (0.20, ((0.4920, 0.4200, 0.5295), (0.5190, 0.5020, 0.5401), (0.0249, 0.0086, 0.0407), (0.4638, 0.3345, 0.6042))),

    (0.05, ((0.4584, 0.2358, 0.5495), (0.5416, 0.3721, 0.5944), (0.1514, 0.0991, 0.1946), (0.3727, 0.3203, 0.4493))),
    (0.05, ((0.4382, 0.3082, 0.6067), (0.5261, 0.4634, 0.6035), (0.1140, 0.0840, 0.1409), (0.3716, 0.3074, 0.4428))),
    (0.10, ((0.5191, 0.3015, 0.5501), (0.5388, 0.4305, 0.6132), (0.1418, 0.0956, 0.1739), (0.3479, 0.2871, 0.4105))),
    # didn't run long enough for reflectivity to develop
    # (0.10, ((0.3698, 0.2778, 0.4008), (0.4432, 0.4070, 0.4733), (0.0790, 0.0669, 0.1164), (0.3646, 0.2847, 0.4419))),
    (0.20, ((0.4902, 0.4147, 0.5155), (0.5341, 0.4760, 0.5780), (0.0269, 0.0127, 0.0422), (0.4124, 0.2874, 0.5553))),
    (0.20, ((0.4646, 0.4240, 0.5290), (0.5233, 0.4832, 0.5900), (0.0250, 0.0123, 0.0751), (0.4121, 0.3187, 0.5336))),
    (0.30, ((0.3527, 0.2446, 0.3690), (0.4434, 0.4279, 0.4474), (0.0023, 0.0011, 0.0031), (0.3467, 0.3050, 0.4099))),
]

if __name__ == '__main__':
    f, (ax1, ax3) = plt.subplots(2, 1, sharex=True)
    f.subplots_adjust(hspace=0)
    offsets = defaultdict(float)

    msize=3
    lwidth=0.7

    # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k'] * 5
    for re_inv, ((r_med, r_min, r_max),
                 (rA_med, rA_min, rA_max),
                 (T_med, T_min, T_max),
                 (w_med, w_min, w_max)) in DATA:
        # Reynolds number for linear mode
        x = (1024 / 10) / re_inv + offsets[re_inv]
        ax1.plot(x, r_med, 'ko', markersize=msize)
        ax1.errorbar(x, r_med, np.array([[r_med - r_min, r_max - r_med]]),
                     ecolor='k', linewidth=lwidth)
        ax1.plot(x + 14, T_med, 'r*', markersize=msize)
        ax1.errorbar(x + 14, T_med, np.array([[T_med - T_min, T_max - T_med]]),
                     ecolor='r', linewidth=lwidth)
        ax1.plot(x + 7, rA_med**2, 'bo', markersize=msize)
        ax1.errorbar(x + 7, rA_med**2, np.array([[rA_med**2 - rA_min**2,
                                              rA_max**2 - rA_med**2]]),
                     ecolor='b', linewidth=lwidth)
        ax3.plot(x, w_med, 'ko', markersize=msize)
        ax3.errorbar(x, w_med, np.array([[w_med - w_min, w_max - w_med]]),
                     ecolor='k', linewidth=lwidth)

        ax1.set_ylim([0, 1])
        ax1.set_yticks([0, 0.3, 0.6])
        ax3.set_ylim([0, 0.7])
        ax3.set_yticks([0, 0.5])
        offsets[re_inv] += 20

    ln1 = mlines.Line2D([], [], color='k', marker='o', markersize=msize,
                        label=r'$\left<\mathcal{R}_S(t)\right>$')
    ln2 = mlines.Line2D([], [], color='k', marker='o', markersize=msize,
                        label=r'$\left<\mathcal{R}_A(t)^2\right>$')
    ln3 = mlines.Line2D([], [], color='r', marker='*', markersize=msize,
                        label=r'$\left<\mathcal{T}_S(t)\right>$')
    ax1.legend(handles=[ln1, ln2, ln3], fontsize=6)


    ax1.set_ylabel(r'$\left<\mathcal{R}_S(t)\right>$')
    # ax2.set_ylabel(r'$\left<\mathcal{R}_A^2(t)\right>$')
    ax3.set_ylabel('Ri')
    ax3.set_xlabel('Re')
    ax = plt.gca()
    # ax.set_xscale('log')
    plt.savefig('agg.png', dpi=400)

