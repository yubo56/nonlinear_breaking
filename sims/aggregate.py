'''
using data outputted from 2d_3_final, 2d_4_fourier on refl/Ri
'''
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

DATA = [
    (0.30, ((0.2672, 0.1594, 0.3145), (0.4387, 0.2583, 0.6056))),
    (0.20, ((0.4652, 0.2961, 0.5378), (0.4638, 0.2690, 0.6856))),
    (0.05, ((0.4103, 0.1838, 0.5486), (0.3727, 0.2944, 0.4946))),
    (0.05, ((0.4393, 0.2670, 0.6216), (0.3716, 0.2763, 0.5213))),
    (0.10, ((0.4195, 0.1315, 0.5542), (0.3479, 0.2555, 0.4847))),
    # (0.10, ((0.2809, 0.2164, 0.3398), (0.3646, 0.2471, 0.5146))),
    (0.20, ((0.4687, 0.3770, 0.5762), (0.4124, 0.2434, 0.6674))),
    (0.20, ((0.4421, 0.3387, 0.6575), (0.4121, 0.2549, 0.6523))),
    (0.30, ((0.3525, 0.2183, 0.3855), (0.3467, 0.2604, 0.4354))),
]

if __name__ == '__main__':
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    f.subplots_adjust(hspace=0)
    for re_inv, ((r_med, r_min, r_max), (w_med, w_min, w_max)) in DATA:
        ax1.scatter(1 / re_inv, r_med)
        ax1.errorbar(1 / re_inv, r_med, np.array([[r_min, r_max]]))
        ax2.scatter(1 / re_inv, w_med)
        ax2.errorbar(1 / re_inv, w_med, np.array([[w_min, w_max]]))

    ax1.set_ylabel(r'$\mathcal{R}_S$')
    ax2.set_xlabel('Re')
    ax2.set_ylabel('Ri')
    plt.savefig('agg.png')

