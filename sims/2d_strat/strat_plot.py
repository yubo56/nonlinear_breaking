#!/usr/bin/env python
"""
plot single run results for strat.py, strat_sommer.py
"""
import sys
import os

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from dedalus.extras.plot_tools import pad_limits

def split_path(path):
    return (
        '/'.join(path.split('/')[ :-1]),
        '.'.join(path.split('/')[-1].split('.')[ :-1])
    )
SAVE_FMT_STR = 't_%d.png'

if __name__ == '__main__':
    matplotlib.rcParams.update({'font.size': 6})
    filenames = sys.argv[1: ]
    plot_vars = ['uz', 'ux', 'rho', 'P']
    n_cols = 2
    n_rows = 2
    assert len(plot_vars) == n_cols * n_rows

    for filename in filenames:
        with h5py.File(filename, mode='r') as dat:
            path, filetitle = split_path(filename)

            sim_times = np.array(dat['tasks']['uz'].dims[0][0])
            xmesh = np.array(dat['tasks']['uz'].dims[1][0])
            zmesh = np.array(dat['tasks']['uz'].dims[2][0])

            for t_idx, sim_time in enumerate(sim_times):
                fig = plt.figure(dpi=200)

                for idx, var in enumerate(plot_vars):
                    axes = fig.add_subplot(n_cols, n_rows, idx + 1, title=var)

                    var_dat = np.array(dat['tasks'][var])
                    p = axes.pcolormesh(xmesh,
                                        zmesh,
                                        np.transpose(var_dat[t_idx]),
                                        vmin=var_dat.min(), vmax=var_dat.max())
                    axes.axis(pad_limits(xmesh, zmesh))
                    fig.colorbar(p, ax=axes)

                fig.suptitle('%s (t=%.2f)' % (filetitle,
                                              sim_time))
                fig.subplots_adjust(hspace=0.2, wspace=0.2)
                savefig = SAVE_FMT_STR % t_idx
                plt.savefig('%s/%s' % (path, savefig))
                print('Saved %s' % savefig)
                plt.close()
            os.system('ffmpeg -y -framerate 10 -i %s/%s %s.mp4' %
                      (path, SAVE_FMT_STR, path.replace('/', '_')))
