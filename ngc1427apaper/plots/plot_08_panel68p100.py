#! /usr/bin/env python
import pickle
from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pynbody
import tqdm
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import ImageGrid
from simulation.luminosity import surface_brightness, color_plot
from simulation.simdata import SIM_NAME_DICT, SPECIAL_DICT, get_traj

from ngc1427apaper.helper import (age_map, cii_map, get_data, get_data_name,
                                  add_cluster_center_velocity_directions, phitheta2rotation, savefig)
from ngc1427apaper.maps import SolvedPreparedSnap

SIM_DICT = {**SIM_NAME_DICT, **SPECIAL_DICT}

# mpl.style.use('MNRAS')

# From matplotlib 3.2 is deprecated. It's also buggy, much better using the native one.
mpl.rcParams['mpl_toolkits.legacy_colorbar'] = False

def sb_im(snap, band, width, resolution, mag_filter):
    sb = surface_brightness(snap, band=band, width=width, resolution=resolution,
                            mag_filter=mag_filter, noplot=True)
    return sb

def stellar_velocity_im(snap, sb, width, resolution,
                        minimum_sb=None, astronomical_convention=False,
                        center_on_mean=False):
    sign = -1 if astronomical_convention else 1
    star_vel = sign * pynbody.plot.sph.image(snap.s, qty='vz', av_z='v_lum_den',
                                  width=width, resolution=resolution, noplot=True)
    star_vel[np.isnan(sb)] = np.nan
    if minimum_sb is not None:
        star_vel[sb > minimum_sb] = np.nan
    if center_on_mean:
        star_vel = star_vel - np.nanmean(star_vel)
    return star_vel


def hi_im(snap, width, resolution):
    return pynbody.plot.sph.image(snap.g,
                                  qty='rho_HI', units='amu cm**-2',
                                  width=width, resolution=resolution, noplot=True)



# def fill_grid(d, grid, vrange, norm, label):
#     for i, ax in enumerate(grid_v):
#         s = maps[i]
#         print(s.which_snap)
#         age = _age_map(s.snap, sb_list[i], width, s.resolution)
#         ax.imshow(age, cmap=cmap_name, extent=extent, vmin=vel_range[0], vmax=vel_range[1], origin='lower')
#         add_cluster_center_velocity_directions(ax, s.orbital_position_rotated, s.orbital_velocity_rotated, color_cluster='k', with_text=False)
#     norm_v = mpl.colors.Normalize(vmin=vel_range[0], vmax=vel_range[1])

def compute_maps(maps, width, mag_filter, cii_filter=1e-9):
    d = defaultdict(list)
    for s in tqdm.tqdm(maps):
        sb = sb_im(s.snap, s.band, width, s.resolution, mag_filter=mag_filter)
        d['sb'].append(sb)
        d['hi'].append(hi_im(s.snap, width, s.resolution))
        d['age'].append(age_map(s.snap, sb, width, s.resolution))
        d['color'].append(color_plot(s.snap, bands=('sdss_g', 'sdss_r'),
                   width=width, resolution=s.resolution,
                   mag_filter=mag_filter,
                   noplot=True,
                   # vmin=-0.3, vmax=0.6,
                   ))
        d['kinematics'].append(stellar_velocity_im(s.snap, sb, width, s.resolution,
            astronomical_convention=True, center_on_mean=True))

        cii = cii_map(s.snap, width, s.resolution)
        cii[cii < cii_filter] = np.nan
        d['cii'].append(cii)
    return d


rows = ('sb', 'color', 'age', 'kinematics')

levels_hi = np.logspace(17, 22, 5)
sb_range = (21, 27.5)
vel_range = (-40, 40)


_titles = {'sb': 'Surface Brightness',
           'age': 'Age',
           'color': "g'-r'",
           'kinematics': 'Kinematics',
           'cii': "[CII]"}

_vranges = {'sb':sb_range,
            'age': (0,8),
           'color': (-0.2, 1),
           'kinematics': (-30, 30),
           'cii': (1e-9, 1e-3)}

_labels = {'sb':r"$\mu_{r'}$",
            'age': 'age (Gyr)',
           'color': "g'-r'",
           'kinematics': 'VLOS (km/s)',
           'cii': r"[CII] (erg s$^{{-1}}$ cm$^{{-2}}$)"}


sim_label = '68p100'
which_snap = 49

n_snaps = 7
spacing = 5
limit = spacing*(n_snaps-1)//2
snaps = tuple(np.linspace(-limit, limit, num=n_snaps, dtype=int) + which_snap+limit)

print(snaps)

n_snap = len(snaps)
maps = list()

dff = get_data('cache_with_iso_and_hi_valid.pkl')

df = dff.query(f'sim == "{sim_label}" & snap == {which_snap} & rp == 77 & vp == -793 & sol == 2 & sign == 1')
print(df)
assert len(df) == 1

sim_name = SIM_DICT[sim_label]
tbl_traj = get_traj(sim_name)
phi, theta = float(df.phi), float(df.theta)
orbital_position = np.array(df[['x', 'y', 'z']])[0]
# print(orbital_position)

# rotation, phi, theta = get_solved_rotation(orbital_position, orbital_velocity, rp, vp, which_sol, sign_of_r)
rotation = phitheta2rotation(phi, theta, orbital_position)

for snap in snaps:
    maps.append(SolvedPreparedSnap(sim_label=sim_label, which_snap=snap, rotation=rotation))


# nrows = 2
nrows = len(rows)

width = 24
extent = (-width/2, width/2, -width/2, width/2)

cmap_name = 'nipy_spectral'

load_pickle = True

pickle_name = 'dd68p100.pkl'
if load_pickle:
    d = pickle.load(open(get_data_name(pickle_name), 'rb'))
    print(f'loaded {pickle_name}...')
else:
    d = compute_maps(maps, width, mag_filter=sb_range[1])
    pickle.dump(d, open(pickle_name, 'wb'))


fig = plt.figure(figsize=(2.5*n_snap, 2.5*nrows), dpi=300)
grids = dict()
for i, r in enumerate(rows):
    print(r)
    grids[r] = ImageGrid(fig, int(f"{nrows}1{i+1}"),  # similar to subplot(142)
                 nrows_ncols=(1, n_snap),
                 axes_pad=0.0,
                 share_all=True,
                 # label_mode="L",
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size='5%',
                 cbar_pad=0.0,
                 cbar_set_cax=True,
                 )



for r in rows:
    g = grids[r]
    vrange = _vranges[r]
    print(r)
    if r == 'cii':
        norm = LogNorm(*vrange)
    else:
        norm = Normalize(*vrange)
    for i, ax in enumerate(g):
        ax.imshow(d[r][i], cmap=cmap_name, extent=extent, norm=norm, origin='lower')

# special ones
xtext, ytext = 0.5, 0.87
for i, ax in enumerate(grids['sb']):
    cs_hi = ax.contour(d['hi'][i], extent=extent,
                       levels=levels_hi, colors='k',
                       linewidths=0.8, linestyles='--', alpha=0.8)
    s = maps[i]
    add_cluster_center_velocity_directions(ax,
                         s.orbital_position_rotated, s.orbital_velocity_rotated,
                         color_cluster='k', with_text=False)
    ax.text(xtext, ytext, rf"$\tau$={s.tau:.2f}, {s.time_since_peri * 1000:.0f} Myr since peri",
            c='k', size='small', weight='normal', transform=ax.transAxes, horizontalalignment='center')

# titles
for qty, g in grids.items():
    g.axes_all[n_snap//2].set_title(_titles[qty])
    for ax in g:
        ax.grid(ls=':')

# colorbars
cmap = plt.get_cmap(cmap_name)
# cmap.set_bad('black')
for qty, g in grids.items():
    if qty == 'cii':
        norm = LogNorm(*_vranges[qty])
    else:
        norm = Normalize(*_vranges[qty])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = g.cbar_axes[0]
    cbar = cbar_ax.colorbar(sm)
    cbar_ax.set_ylabel(_labels[qty])


# remove internal ticks. Use dict order
for i, g in enumerate(grids.values()):
    g.axes_column[0][0].set_ylabel('y (kpc)')

    if i == len(grids)-1:
        for ax in g.axes_row[-1]:
            ax.set_xlabel('x (kpc)')
        continue
    for ax in g:
        ax.set_xticklabels([])

# fig.subplots_adjust(wspace=0, hspace=0)


# This affects all axes as share_all = True.
# grid.axes_llc.set_xticks([-2, 0, 2])
# grid.axes_llc.set_yticks([-2, 0, 2])
# grid.axes_all.set_xlabel('x/kpc')
# grid.axes_all.set_ylabel('y/kpc')

file_stem = 'panel68p100'
savefig(fig, file_stem, '.png', dpi=300)
# savefig(fig, file_stem, '.pdf')
plt.show()
