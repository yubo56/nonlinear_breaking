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
    (0.05, ((0.5169, 0.4096, 0.6131), (0.5571, 0.4914, 0.6078), (0.1089, 0.0768, 0.1456), (0.3716, 0.3074, 0.4428))),
    (0.10, ((0.5167, 0.4397, 0.5675), (0.6195, 0.5759, 0.6609), (0.1911, 0.1428, 0.2289), (0.3829, 0.3202, 0.4596))),
    (0.20, ((0.4626, 0.4088, 0.5010), (0.5487, 0.4951, 0.5867), (0.0340, 0.0207, 0.0456), (0.4124, 0.2874, 0.5553))),
    (0.30, ((0.3475, 0.2300, 0.3747), (0.4441, 0.4382, 0.4462), (0.0022, 0.0009, 0.0024), (0.3467, 0.3050, 0.4099))),
    (0.50, ((0.0723, 0.0606, 0.1253), (0.2085, 0.1766, 0.2216), (0.0000, 0.0000, 0.0000), (1.2511, 0.7374, 5.8832))),
    (0.70, ((0.0677, 0.0623, 0.0883), (0.0849, 0.0790, 0.0953), (0.0000, 0.0000, 0.0000), (50.3429, 8.8182, 98.6669))),
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
    plt.savefig('agg.png', dpi=600)

