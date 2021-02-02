#! /usr/bin/env python
import matplotlib.pyplot as plt

from ngc1427apaper.globals import *
from ngc1427apaper.helper import *


def filter_out(df, tol, iso=0, with_retro=True, real_alpha=REAL_ALPHA, real_beta=REAL_BETA):
    rx, ry = np.cos(np.pi/3), np.sin(np.pi/3)
    if with_retro:
        l = df.query(f'{real_alpha-tol}<alpha{iso}<{real_alpha+tol} & {real_beta-tol}<beta{iso}<{real_beta+tol} &\
                   (vx_rot * {rx} + vy_rot * {ry})>0 & hi_max > 1e15').copy()
    else:
        l = df.query(f'{real_alpha-tol}<alpha{iso}<{real_alpha+tol} & {real_beta-tol}<beta{iso}<{real_beta+tol} &\
                   (vx_rot * {rx} + vy_rot * {ry})>0 & not sim.str.endswith("r") & hi_max > 1e15').copy()
    l.loc[:,'tol'] = tol
    l.loc[:,'iso'] = isophote_target[iso]
    # print(l)
    return l

def plot_tp(df, ax, max_val=None, **kwargs):
    bins = np.linspace(-0.35, 0.35, 25)

    what='tp'

    for i in (1,2):
        df.query(f'nearest_peri == {i}')[f'tp{i}'].hist(ax=ax, legend=False, bins=bins, alpha=0.6)

    # x = (df.query(f'nearest_peri == 1')[f'tp1'], df.query(f'nearest_peri == 2')[f'tp2'])
    # ax.hist((x[0], x[1]),
    #         bins=bins,
    #         alpha=0.6,
    #         # color=('C1', 'C0'),
    #         **kwargs)
    if what == 't_period':
        ax.set_xlabel(r'$\tau$')
    else:
        ax.set_xlabel(r'Gyr')

    if max_val is not None:
        ax.set_ylim(0, max_val)
    ax.grid(ls=':')
    ax.set_ylabel('# oriented snaps\n'rf'($\delta$={df.iloc[0]["tol"]}deg)')


if __name__ == '__main__':
    dff = get_data('selected_with_iso_and_morph_and_data.pkl')
    df20 = filter_out(dff, 20, iso=0), filter_out(dff, 20, iso=1), filter_out(dff, 20, iso=2)

    nrows = 1
    from mpl_toolkits.axes_grid1 import ImageGrid
    fig = plt.figure(figsize=(11, 3*nrows))
    grid = ImageGrid(fig, 111,
                 nrows_ncols=(nrows, 3),
                 axes_pad=0.0,
                 share_all=True,
                 label_mode="L",
                 aspect=False, # If true ImageGrid does not listen to figsize parameters! It does not work...
                 )

    colors = 'r','g','b'
    for _df, row in zip((df20,), range(nrows)):
        for iso in (0,1,2):
            plot_tp(_df[iso], ax=grid.axes_row[row][iso], max_val=60)

    grid.axes_all[2].legend(['second pericenter passage', 'first pericenter passage'])

    for i, ax in enumerate(grid.axes_row[0]):
        ax.set_title(fr'{isophote_target[i]} mag arcsec$^{{-2}}$')

    # Savefig
    file_stem = f"one_tol_bins0.35-25"
    savefig(fig, file_stem, '.png')
    savefig(fig, file_stem, '.pdf')

    plt.show()
