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

DATA = [
    (0.70, ((0.1182, 0.0963, 0.1293), (0.1416, 0.1220, 0.1560), (0.9475, 0.8792, 1.1203))),
    (0.50, ((0.1768, 0.1491, 0.1943), (0.2478, 0.2159, 0.2763), (0.5811, 0.4863, 0.6789))),
    (0.30, ((0.2672, 0.1594, 0.3145), (0.4193, 0.3933, 0.4364), (0.4387, 0.2583, 0.6056))),
    (0.20, ((0.4652, 0.2961, 0.5378), (0.5190, 0.4977, 0.5555), (0.4638, 0.2690, 0.6856))),
    (0.05, ((0.4103, 0.1838, 0.5486), (0.5416, 0.3546, 0.6245), (0.3727, 0.2944, 0.4946))),
    (0.05, ((0.4393, 0.2670, 0.6216), (0.5261, 0.4435, 0.6283), (0.3716, 0.2763, 0.5213))),
    (0.10, ((0.4195, 0.1315, 0.5542), (0.5388, 0.3308, 0.6575), (0.3479, 0.2555, 0.4847))),
    # didn't run long enough for reflectivity to develop
    # (0.10, ((0.2809, 0.2164, 0.3398), (0.4432, 0.3634, 0.4951), (0.3646, 0.2471, 0.5146))),
    (0.20, ((0.4687, 0.3770, 0.5762), (0.5341, 0.4573, 0.6030), (0.4124, 0.2434, 0.6674))),
    (0.20, ((0.4421, 0.3387, 0.6575), (0.5233, 0.4624, 0.6154), (0.4121, 0.2549, 0.6523))),
    (0.30, ((0.3525, 0.2183, 0.3855), (0.4434, 0.4154, 0.4564), (0.3467, 0.2604, 0.4354))),
]

if __name__ == '__main__':
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    f.subplots_adjust(hspace=0)
    offsets = defaultdict(float)

    for re_inv, ((r_med, r_min, r_max), (rA_med, rA_min, rA_max), (w_med, w_min, w_max)) in DATA:
        x = 1 / re_inv + offsets[re_inv]
        ax1.scatter(x, r_med)
        ax1.errorbar(x, r_med, np.array([[r_min, r_max]]))
        ax2.scatter(x, rA_med**2)
        ax2.errorbar(x, rA_med**2, np.array([[rA_min**2, rA_max**2]]))
        ax3.scatter(x, w_med)
        ax3.errorbar(x, w_med, np.array([[w_min, w_max]]))
        offsets[re_inv] += 0.2

    ax1.set_ylabel(r'$\mathcal{R}_S$')
    ax2.set_ylabel(r'$\mathcal{R}_A^2$')
    ax3.set_ylabel('Ri')
    ax3.set_xlabel('Re')
    ax = plt.gca()
    # ax.set_xscale('log')
    plt.savefig('agg.png', dpi=400)

