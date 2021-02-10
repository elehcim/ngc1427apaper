#! /usr/bin/env python
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.stats import gaussian_kde

from ngc1427apaper.globals import isophote_target
from ngc1427apaper.helper import REAL_BETA, REAL_ALPHA, get_data, savefig


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

def plot_hist2d(df, ax, max_val=None, nearest_peri=1, **kwargs):
    bins = None

    iso = isophote_target.index(df['iso'].iloc[0])
    # df.query(f'nearest_peri == {nearest_peri}')[f'tp{nearest_peri}'].hist(
    # what = f'tp{nearest_peri}'
    # what = f'e{iso}'
    # what = f'mass_star'
    # what = f'dc{iso}'
    # what = f'temp_host'
    # what = f'p_host'
    # what = f'RPS'
    what = f'asymmetry'

    # if what == 'mass_star':
    #     x = (np.log10(df.query(f'nearest_peri == 1')[what]), np.log10(df.query(f'nearest_peri == 2')[what]))
    # else:
    #     x = (df.query(f'nearest_peri == 1')[what], df.query(f'nearest_peri == 2')[what])
    # what =
    df1 = df.query(f'nearest_peri == 1')
    x = df1['tp1']
    # x = df1['asymmetry']
    y = df1[what]
    im, _, _, _ = ax.hist2d(x, y,
            range=[[-0.27, 0.27], [0, 1.25]],
            **kwargs)

    # kde
    nbins = 50 #kwargs.get('bins', 10)
    df1 = df.query(f'nearest_peri == 1 & tp1 < 0.1')[['tp1', what]].drop_duplicates()
    x = df1['tp1']
    # x = df1['asymmetry']
    y = df1[what]
    data = np.array([x,y])
    k = gaussian_kde(data)
    xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    g = np.vstack([xi.flatten(), yi.flatten()])
    zi = k(g)

    # plot a density
    # ax.set_title('Calculate Gaussian KDE')
    # ax.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.BuGn_r)
    ax.contour(xi, yi, zi.reshape(xi.shape), 4)
    # ax.clabel(CS, fmt='%.1f')


    ax.set_xlabel(r'$t-t_p$ (Gyr)')
    ax.set_ylabel(f'A')

    # nbins = kwargs.get('bins', 10)
    # data = np.array([x,y])
    # k = gaussian_kde(data.T)
    # xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    # zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    return im


if __name__ == '__main__':
    dff = get_data('selected_with_iso_and_morph_and_data.pkl')
    df = filter_out(dff, 20, iso=0), filter_out(dff, 20, iso=1), filter_out(dff, 20, iso=2)

    nrows = 1
    ncols = len(df)

    fig = plt.figure(figsize=(5*ncols, 5*nrows))
    grid = ImageGrid(fig, 111,
                 nrows_ncols=(nrows, ncols),
                 axes_pad=0.0,
                 share_all=True,
                 label_mode="L",
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size='5%',
                 cbar_pad=0.0,
                 cbar_set_cax=True,
                 aspect=False, # If true ImageGrid does not listen to figsize parameters! It does not work...
                 )

    cmap_name = 'viridis'
    bins = 20
    lim = list()
    for _df, row in zip((df,), range(nrows)):
        for iso in (0,1,2):
            lim.append(plot_hist2d(_df[iso], ax=grid.axes_row[row][iso], density=False, cmap=cmap_name, bins=bins))

    # for l in lim:
    #     print(np.max(l))
    vmax = np.max(lim)
    cmap = matplotlib.cm.get_cmap(cmap_name)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])

    cax = grid.cbar_axes[0]
    cbar = plt.colorbar(mappable, cax=cax)
    cax.set_ylabel('# oriented snaps\n'rf'($\delta$={_df[iso].iloc[0]["tol"]}deg)')
    # from https://stackoverflow.com/a/34880501
    # This is also interesting: https://matplotlib.org/3.3.1/gallery/ticks_and_spines/tick-formatters.html
    from matplotlib import ticker
    cax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    # cax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    # cax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=5))

    for i, ax in enumerate(grid.axes_row[0]):
        ax.set_title(fr'{isophote_target[i]} mag arcsec$^{{-2}}$')

    for ax in grid.axes_row[0]:
        ax.axhline(y=0.23, color='red', ls=':', linewidth=0.8)

    # grid.axes_row[0][0].text((-0.24, 0.24), 'A=0.23')

    # Savefig
    file_stem = f"hist2d_tp_A_bins{bins}_kde"
    savefig(fig, file_stem, '.png', dpi=150)
    savefig(fig, file_stem, '.pdf', dpi=150)


    # Try KDE
    # fig, ax = plt.subplots()

    # nbins = bins#kwargs.get('bins', 10)
    # df1 = df[1].query(f'nearest_peri == 1 & tp1<0.4')[['tp1', 'dc1']].drop_duplicates()
    # x = df1['tp1']
    # # x = df1['asymmetry']
    # y = df1[f'dc1']
    # data = np.array([x,y])
    # k = gaussian_kde(data)
    # xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    # g = np.vstack([xi.flatten(), yi.flatten()])
    # zi = k(g)

    # # plot a density
    # ax.set_title('Calculate Gaussian KDE')
    # ax.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.BuGn_r)
    # ax.contour(xi, yi, zi.reshape(xi.shape))

    # plt.show()
