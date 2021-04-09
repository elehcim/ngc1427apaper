#! /usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid

from ngc1427apaper.globals import isophote_target
from ngc1427apaper.helper import REAL_ALPHA, REAL_BETA, get_data, savefig

bin_limit = 90
nbins = 25

def filter_out(df, tol, iso=0, with_retro=False, real_alpha=REAL_ALPHA, real_beta=REAL_BETA):
    rx, ry = np.cos(np.pi/3), np.sin(np.pi/3)
    if with_retro:
        l = df.query(f'{real_alpha-tol}<alpha{iso}<{real_alpha+tol} & {real_beta-tol}<beta{iso}<{real_beta+tol} &\
                   (vx_rot * {rx} + vy_rot * {ry})>0 & hi_max > 1e15'
                   ).copy()
    else:
        l = df.query(f'{real_alpha-tol}<alpha{iso}<{real_alpha+tol} & {real_beta-tol}<beta{iso}<{real_beta+tol} &\
                    (vx_rot * {rx} + vy_rot * {ry})>0 & not sim.str.endswith("r") & hi_max > 1e15'
                    ).copy()
    l.loc[:,'tol'] = tol
    l.loc[:,'iso'] = isophote_target[iso]
    # print(l)
    return l

def plot_tp(df, ax, max_val=None, **kwargs):
    bins = np.linspace(0, bin_limit, nbins)

    what = 'vel_angle_abs'

    print(kwargs['color'])
    df[what].hist(ax=ax, legend=False, range=(0,90),
     bins=bins,
      alpha=0.6,
      **kwargs)
    # ax2=ax.twinx()
    # df[what].plot.kde(ax=ax2)
    # ax2.set_xlim((0,90))
    # ax2.axis('off')
    # x = (df.query(f'nearest_peri == 1')[f'tp1'], df.query(f'nearest_peri == 2')[f'tp2'])
    # ax.hist((x[0], x[1]),
    #         bins=bins,
    #         alpha=0.6,
    #         # color=('C1', 'C0'),
    #         **kwargs)
    ax.set_xlabel(r'$\gamma$ (deg)')

    # if max_val is not None:
    #     ax.set_ylim(0, max_val)
    ax.grid(ls=':')
    ax.set_ylabel('# oriented snaps')


if __name__ == '__main__':
    dff = get_data('selected_with_multi_iso_and_morph_and_data.pkl')
    dff['vxy_rot'] = np.linalg.norm([dff.vx_rot, dff.vy_rot], axis=0)
    dff['vel_angle'] = np.arctan2(dff.vxy_rot, -dff.vz_rot)*180/np.pi
    dff['vel_angle_abs'] = np.abs(dff['vel_angle'])
    # tol = 20
    iso = 1
    df = filter_out(dff, 20, iso=iso), filter_out(dff, 15, iso=iso), filter_out(dff, 10, iso=iso)

    nrows = 1
    ncols = 1
    fig = plt.figure(figsize=(4, 3*nrows))
    grid = ImageGrid(fig, 111,
                 nrows_ncols=(nrows, ncols),
                 axes_pad=0.0,
                 share_all=True,
                 label_mode="L",
                 aspect=False, # If true ImageGrid does not listen to figsize parameters! It does not work...
                 )

    colors = 'C0', 'C1', 'C2'
    for _df, c in zip(df, colors):
        plot_tp(_df, ax=grid.axes_row[0][0], max_val=60, color=c)

    grid.axes_all[0].legend([r'$\delta$=20 deg', r'$\delta$=15 deg', r'$\delta$=10 deg'])

    # for i, ax in enumerate(grid.axes_row[0]):
        # ax.set_title(fr'{isophote_target[1]} mag arcsec$^{{-2}}$'))

    # Savefig
    file_stem = f"multi_tol_vel_angle_bins_gauss{bin_limit}_{nbins}"
    # savefig(fig, file_stem, '.png')
    savefig(fig, file_stem, '.pdf')

    # plt.show()
