#! /usr/bin/env python
import matplotlib.pyplot as plt

from ngc1427apaper.globals import *
from ngc1427apaper.helper import *


def filter_out(df, tol, iso=1, with_retro=True, real_alpha=REAL_ALPHA, real_beta=REAL_BETA):
    rx, ry = np.cos(np.pi/3), np.sin(np.pi/3)
    if with_retro:
        l = df.query(f'{real_alpha-tol}<alpha{iso}<{real_alpha+tol} & {real_beta-tol}<beta{iso}<{real_beta+tol} &\
                   (vx_rot * {rx} + vy_rot * {ry})>0 & hi_max > 1e15 & rp==137 & vp==-693').copy()
    else:
        l = df.query(f'{real_alpha-tol}<alpha{iso}<{real_alpha+tol} & {real_beta-tol}<beta{iso}<{real_beta+tol} &\
                   (vx_rot * {rx} + vy_rot * {ry})>0 & not sim.str.endswith("r") & hi_max > 1e15').copy()
    l.loc[:,'tol'] = tol
    l.loc[:,'iso'] = isophote_target[iso]
    # print(l)
    return l


def plot_1_passage(df, axs=None, max_val=None, nearest_peri=1):
    # groups = df.query('peri != 300').groupby('peri')
    bins = np.linspace(-0.35, 0.35, 16)
    groups = df.groupby('peri')

    for i, (peri, g) in enumerate(groups):
        g.query(f'nearest_peri == {nearest_peri}')[f'tp{nearest_peri}'].hist(ax=axs[i], legend=False, bins=bins, alpha=0.6)

        axs[i].set_xlabel(r'$t-t_p$ (Gyr)')

        if max_val is not None:
            axs[i].set_ylim(0, max_val)


    axs[0].set_ylabel('# oriented snaps\n'rf'($\delta$={df.iloc[0]["tol"]}deg)')


if __name__ == '__main__':
    dff = get_data('selected_with_iso_and_morph_and_data.pkl')
    tol = 10
    df10 = filter_out(dff, tol)
    df15 = filter_out(dff, 15)
    df20 = filter_out(dff, 20)

    from mpl_toolkits.axes_grid1 import ImageGrid
    fig = plt.figure(figsize=(15, 8))
    grid = ImageGrid(fig, 111,
                 nrows_ncols=(3, 5),
                 axes_pad=0.0,
                 share_all=True,
                 label_mode="L",
                 aspect=False, # If true ImageGrid does not listen to figsize parameters! It does not work...
                 )

    plot_1_passage(df20, axs=grid.axes_row[0], max_val=10)
    plot_1_passage(df20, axs=grid.axes_row[0], nearest_peri=2)
    plot_1_passage(df15, axs=grid.axes_row[1])
    plot_1_passage(df15, axs=grid.axes_row[1], nearest_peri=2)
    plot_1_passage(df10, axs=grid.axes_row[2])
    plot_1_passage(df10, axs=grid.axes_row[2], nearest_peri=2)
    grid.axes_all[2].legend(['first pericenter passage', 'second pericenter passage'], fontsize='x-small')

    for ax, (peri, _) in zip(grid.axes_row[0], df20.groupby('peri')):
        ax.set_title('Pericenter: '+str(peri) + ' kpc')

    # Savefig
    file_stem = f"histo_iso26.5_bins0.35-16"
    savefig(fig, file_stem, '.png')
    savefig(fig, file_stem, '.pdf')

    plt.show()
