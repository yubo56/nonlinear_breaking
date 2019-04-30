'''
using data outputted from 2d_3_final, 2d_4_fourier on refl/Ri
'''
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=12)
import matplotlib.lines as mlines

DATA = [
    (0.70, ((0.1129, 0.0950, 0.1267), (0.1416, 0.1220, 0.1560), (0.0001, 0.0000, 0.0006), (0.4834, 0.4486, 0.5716))),
    (0.50, ((0.1737, 0.1541, 0.1919), (0.2478, 0.2159, 0.2763), (0.0040, 0.0017, 0.0063), (0.2965, 0.2481, 0.3464))),
    (0.30, ((0.2659, 0.2046, 0.3090), (0.4193, 0.4122, 0.4270), (0.0000, 0.0000, 0.0002), (0.4387, 0.3322, 0.5149))),
    (0.20, ((0.4920, 0.4200, 0.5295), (0.5190, 0.5020, 0.5401), (0.0249, 0.0086, 0.0407), (0.4638, 0.3345, 0.6042))),

    (0.05, ((0.4869, 0.3428, 0.5560), (0.5632, 0.4939, 0.5996), (0.1528, 0.0863, 0.1885), (0.3727, 0.3203, 0.4493))),
    (0.05, ((0.5169, 0.4096, 0.6131), (0.5571, 0.4914, 0.6078), (0.1089, 0.0768, 0.1456), (0.3716, 0.3074, 0.4428))),
    (0.10, ((0.4914, 0.2686, 0.5305), (0.5287, 0.3905, 0.6236), (0.1538, 0.1016, 0.1787), (0.3479, 0.2871, 0.4105))),
    (0.10, ((0.5220, 0.3271, 0.6194), (0.5603, 0.4294, 0.5899), (0.1274, 0.1000, 0.1805), (0.3627, 0.2888, 0.4312))),
    (0.20, ((0.4626, 0.4088, 0.5010), (0.5487, 0.4951, 0.5867), (0.0340, 0.0207, 0.0456), (0.4124, 0.2874, 0.5553))),
    (0.30, ((0.3475, 0.2300, 0.3747), (0.4441, 0.4382, 0.4462), (0.0022, 0.0009, 0.0024), (0.3467, 0.3050, 0.4099))),
]

if __name__ == '__main__':
    f, (ax1, ax3) = plt.subplots(2, 1, sharex=True)
    f.subplots_adjust(hspace=0)
    offsets = defaultdict(float)

    msize=3
    lwidth=0.7

    for re_inv, ((r_med, r_min, r_max),
                 (rA_med, rA_min, rA_max),
                 (T_med, T_min, T_max),
                 (w_med, w_min, w_max)) in DATA:
        # Reynolds number for linear mode
        print(1024 / 10 / re_inv)
        x = (1024 / 10) / re_inv + offsets[re_inv]
        ax1.plot(x, r_med, 'ko', markersize=msize)
        ax1.errorbar(x, r_med,
                     [[r_med - r_min], [r_max - r_med]],
                     ecolor='k', linewidth=lwidth)
        ax1.plot(x + 14, T_med, 'r*', markersize=msize)
        ax1.errorbar(x + 14, T_med,
                     [[T_med - T_min], [T_max - T_med]],
                     ecolor='r', linewidth=lwidth)
        ax1.plot(x + 7, rA_med**2, 'bo', markersize=msize)
        ax1.errorbar(x + 7, rA_med**2,
                     [[rA_med**2 - rA_min**2], [rA_max**2 - rA_med**2]],
                     ecolor='b', linewidth=lwidth)
        ax3.plot(x, w_med, 'ko', markersize=msize)
        ax3.errorbar(x, w_med, np.array([[w_med - w_min, w_max - w_med]]),
                     ecolor='k', linewidth=lwidth)

        ax1.set_ylim([0, 0.8])
        ax1.set_yticks([0, 0.35, 0.7])
        ax3.set_ylim([0, 0.7])
        ax3.set_yticks([0, 0.5])
        offsets[re_inv] += 20

    ln1 = mlines.Line2D([], [], color='k', marker='o', markersize=msize,
                        label=r'$\left<\mathcal{R}_S(t)\right>$')
    ln2 = mlines.Line2D([], [], color='b', marker='o', markersize=msize,
                        label=r'$\left<\mathcal{R}_A(t)^2\right>$')
    ln3 = mlines.Line2D([], [], color='r', marker='*', markersize=msize,
                        label=r'$\left<\mathcal{T}_S(t)\right>$')
    ax1.legend(handles=[ln1, ln2, ln3], fontsize=12, loc='upper left')


    ax1.set_ylabel(r'$\left<\mathcal{R}_S(t)\right>$')
    ax3.set_ylabel('Ri')
    ax3.set_xlabel('Re')
    ax = plt.gca()
    plt.savefig('agg.png', dpi=400)

