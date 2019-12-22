'''
using data outputted from 2d_3_final, 2d_4_fourier on refl/Ri
'''
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)
import matplotlib.lines as mlines

DATA = [
    (0.05, ((0.5240, 0.4152, 0.6278), (0.5376, 0.4998, 0.5687), (0.1148, 0.0769, 0.1376), (0.3716, 0.3074, 0.4428))),
    (0.10, ((0.5336, 0.4833, 0.5732), (0.5944, 0.5633, 0.6366), (0.1795, 0.1514, 0.2137), (0.3829, 0.3202, 0.4596))),
    (0.20, ((0.4319, 0.4000, 0.4758), (0.5555, 0.5100, 0.6100), (0.0310, 0.0243, 0.0453), (0.4124, 0.2874, 0.5553))),
    (0.30, ((0.2625, 0.2049, 0.3638), (0.4667, 0.4257, 0.4917), (0.0022, 0.0006, 0.0026), (0.3467, 0.3050, 0.4099))),
    (0.50, ((0.0801, 0.0540, 0.0987), (0.2138, 0.1740, 0.2277), (0.0000, 0.0000, 0.0000), (1.2511, 0.7374, 5.8832))),
    (0.70, ((0.0707, 0.0599, 0.0811), (0.0866, 0.0787, 0.0971), (0.0000, 0.0000, 0.0000), (50.3429, 8.8182, 98.6669))),
]

if __name__ == '__main__':
    f, (ax1, ax3) = plt.subplots(2, 1, figsize=(6, 7),
                                 gridspec_kw={'height_ratios': [2, 1]},
                                 sharex=True)
    ax1.set_xscale('log')
    offsets = defaultdict(float)

    msize=5
    lwidth=1.8
    offset_mult = 3

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
        ax1.plot(x + 2 * offset_mult, T_med, 'ro', markersize=msize)
        ax1.errorbar(x + 2 * offset_mult, T_med,
                     [[T_med - T_min], [T_max - T_med]],
                     ecolor='r', linewidth=lwidth)
        ax1.plot(x + offset_mult, rA_med**2, 'bo', markersize=msize)
        ax1.errorbar(x + offset_mult, rA_med**2,
                     [[rA_med**2 - rA_min**2], [rA_max**2 - rA_med**2]],
                     ecolor='b', linewidth=lwidth)

        ax1.set_ylim([0, 0.8])
        ax1.set_yticks([0, 0.35, 0.7])
        ax3.plot(x, w_med, 'ko', markersize=msize)
        ax3.errorbar(x, w_med,
                     [[w_med - w_min], [w_max - w_med]],
                     ecolor='k', linewidth=lwidth)
        offsets[re_inv] += offset_mult * 3
    ax3.set_xlim([100, 2200])
    ax3.set_ylim([0, 1.5])
    ax3.set_yticks([0, 0.5, 1])
    ax3.set_xticks([100, 1000])

    ln1 = mlines.Line2D([], [], color='b', marker='o', markersize=msize,
                        label=r'$\left<\mathcal{R}_A(t)^2\right>$')
    ln2 = mlines.Line2D([], [], color='k', marker='o', markersize=msize,
                        label=r'$\left<\hat{F}_r\right>$')
    ln3 = mlines.Line2D([], [], color='r', marker='*', markersize=msize,
                        label=r'$\left<\hat{F}_s\right>$')
    ax1.legend(handles=[ln1, ln2, ln3], fontsize=14, loc='upper left')

    ax3.set_xlabel('Re')
    ax3.set_ylabel('Ri')
    f.subplots_adjust(hspace=0.1)
    plt.savefig('agg.png', dpi=400)
    plt.clf()
