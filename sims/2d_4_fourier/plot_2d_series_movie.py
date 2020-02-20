"""
Plot planes from joint analysis files.

Usage:
    plot_2d_series_movie.py <files>... [--output=<dir>] [--write=<num>]

Options:
    --output=<dir>  Output directory [default: ./frames]
    --write=<num>   Write number offset [default: 0]
"""

# rm -f frames/*.png && python plot_2d_series_movie.py --output frames snapshots_yubo_nu1_vhres/*.h5
# ffmpeg -framerate 12 -pattern_type glob -i 'yubo_*.png' test.mp4

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
from dedalus.extras import plot_tools
import re

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)

# critical horizontal velocity
UX_C = 0.15394489478851475
SIGMA = 0.078
Z0 = 2
Z_BOT = 0.3
Z_TOP = 9.5

# save two versions of these indicies, one version to put in paper (times are
# labeled)
PLOT_IDXS = [231, 50, 86, 451]
PLOT_PREFIXES = ['(i)', '(ii)', '(iii)', '(iv)']

def main(filename, start, count, output, write_start):
    """Save plot of specified tasks for given range of analysis writes."""

    # Plot settings
    tasks = ['ux','uz']
    scale = 2.5
    dpi = 300
    # savename_func = lambda write: 'yubo_{:06}.png'.format(write+write_start)
    def savename_func(suffix_num):
        return 'yubo_{:06}'.format(suffix_num)
    # Layout
    nrows, ncols = 1, 2
    image = plot_tools.Box(1, 2)
    pad = plot_tools.Frame(0.2, 0.2, 0.1, 0.1)
    margin = plot_tools.Frame(0.15, 0.15, 0.1, 0.1)

    # Create multifigure
    mfig = plot_tools.MultiFigure(nrows, ncols, image, pad, margin, scale)
    fig = mfig.figure
    # Plot writes
    with h5py.File(filename, mode='r') as f:
        for index in range(start, start+count):
            paxes0 = None # use this to overplot text
            for n, task in enumerate(tasks):
                # Build subfigure axes
                i, j = divmod(n, ncols)
                axes = mfig.add_axes(i, j, [0, 0, 1, 1])
                # Call 3D plotting helper, slicing in time
                dset = f['tasks'][task]
                if n==0: #ux
                  clim = [-1.2, 1.2]
                  title = r'$u_x / \overline{U}_c$'
                  func = lambda xmesh, ymesh, data: (xmesh, ymesh, data / UX_C)
                elif n==1: #U
                  clim = [-0.1, 0.1]
                  title = r'$\Upsilon \equiv \ln (\rho / \overline{\rho})$'
                  func = None
                paxes, _ = plot_tools.plot_bot_3d(dset, 0, index, axes=axes, title=title,
                                       even_scale=True,clim=clim, func=func)
                if paxes0 is None:
                    paxes0 = paxes
                # shade in forcing + damping zones
                x = paxes.get_xlim()
                plot_zbot, plot_ztop = paxes.get_ylim()
                paxes.fill_between(x, [plot_zbot, plot_zbot], [Z_BOT, Z_BOT],
                                   alpha=0.3, color='grey')
                paxes.fill_between(x, [Z_TOP, Z_TOP], [plot_ztop, plot_ztop],
                                   alpha=0.3, color='grey')
                paxes.fill_between(x,
                                   [Z0 + 3 * SIGMA, Z0 + 3 * SIGMA],
                                   [Z0 - 3 * SIGMA, Z0 - 3 * SIGMA],
                                   alpha=0.5, color='g')
            # Add time annotation
            sim_time = f['scales/sim_time'][index]
            time_str = '$t=%.1f/N$' % sim_time
            annotation = paxes0.text(0.1, 10.8, time_str, fontsize=14)

            # regex parse out filename number, 10 writes per f
            num_per_file = 10
            write = f['scales/write_number'][index]
            write_num = (write - 1) % num_per_file
            filenum = int(re.match('.*s([\d]+)\.h5', filename)[1])
            suffix_num = (filenum - 1) * num_per_file + write_num
            if suffix_num > 201:
                suffix_num -= 9 # hard coded, one of the restores is misnumbered
            print(suffix_num)

            # Save figure
            savename = savename_func(suffix_num)
            savepath = output.joinpath(savename)
            fig.savefig(str(savepath), dpi=dpi)

            # if in PLOT_IDXS, change overplotted text and plot separately
            try:
                fig_idx = PLOT_IDXS.index(suffix_num)
                annotation.remove()
                print('Adding extra annotation')
                overlay = '%s %s' % (PLOT_PREFIXES[fig_idx], time_str)
                paxes0.text(0.1, 10.8, overlay, fontsize=14)
            except ValueError:
                # suffix_num not in PLOT_IDXS, continue
                continue

            fig.savefig(str(savepath) + '_labeled', dpi=dpi)
            fig.clear()
    plt.close(fig)


if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)

    output_path = pathlib.Path(args['--output']).absolute()
    write_start = int(args['--write'])
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    post.visit_writes(args['<files>'], main, output=output_path, write_start=write_start)

