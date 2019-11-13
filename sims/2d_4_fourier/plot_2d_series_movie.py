"""
rm -f frames/*.png && python plot_2d_series_movie.py testo/*.h5
ffmpeg -framerate 12 -pattern_type glob -i 'yubo_*.png' test.mp4
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
from dedalus.extras import plot_tools
import re

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=12)

def main(filename, start, count, output, write_start):
    """Save plot of specified tasks for given range of analysis writes."""

    # Plot settings
    tasks = ['ux','uz']
    scale = 2.5
    dpi = 300
    title_func = lambda sim_time: r'$t = {:.3f}$'.format(sim_time)
    # savename_func = lambda write: 'yubo_{:06}.png'.format(write+write_start)
    def savename_func(write):
        # regex parse out filename number, 10 writes per file
        num_per_file = 10
        write_num = (write - 1) % num_per_file
        filenum = int(re.match('.*s([\d]+)\.h5', filename)[1])
        suffix_num = (filenum - 1) * num_per_file + write_num
        if suffix_num > 201:
            suffix_num -= 9 # hard coded, one of the restores is misnumbered
        print(suffix_num)
        return 'yubo_{:06}.png'.format(suffix_num)
    # Layout
    nrows, ncols = 1, 2
    image = plot_tools.Box(1, 2)
    pad = plot_tools.Frame(0.2, 0.2, 0.1, 0.1)
    margin = plot_tools.Frame(0.3, 0.2, 0.1, 0.1)

    # Create multifigure
    mfig = plot_tools.MultiFigure(nrows, ncols, image, pad, margin, scale)
    fig = mfig.figure
    # Plot writes
    with h5py.File(filename, mode='r') as file:
        for index in range(start, start+count):
            for n, task in enumerate(tasks):
                # Build subfigure axes
                i, j = divmod(n, ncols)
                axes = mfig.add_axes(i, j, [0, 0, 1, 1])
                # Call 3D plotting helper, slicing in time
                dset = file['tasks'][task]
                if n==0: #ux
                  clim = [-0.25, 0.25]
                  title = r'$u_x$'
                elif n==1: #uz
                  clim = [-0.1, 0.1]
                  title = r'$\Upsilon$'
                plot_tools.plot_bot_3d(dset, 0, index, axes=axes, title=title, even_scale=True,clim=clim)
            # Add time title
            title = title_func(file['scales/sim_time'][index])
            title_height = 1 - 0.5 * mfig.margin.top / mfig.fig.y
            fig.suptitle(title, y=title_height)
            # Save figure
            savename = savename_func(file['scales/write_number'][index])
            savepath = output.joinpath(savename)
            fig.savefig(str(savepath), dpi=dpi)
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

