import os
import platform
from functools import lru_cache

import numpy as np
import pandas as pd
import pynbody
import streamlit as st
from astropy.convolution import convolve, Gaussian2DKernel
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.spatial.transform import Rotation as R
from simulation.derotate_simulation import derotate_pos_and_vel
from simulation.luminosity import surface_brightness
from simulation.util import get_pivot, get_quat, get_omega_mb

from .projection_angles import Problem4

INSET_DIM = dict(width=0.7, height=0.7)

normify = lambda v, fac=1: v/np.linalg.norm(v)/fac

@lru_cache(32)
def cached_get_quat(sim_name):
    return get_quat(sim_name)

@lru_cache(32)
def cached_get_omega_mb(sim_name):
    return get_omega_mb(sim_name)

@lru_cache(1024)
def get_correct_derotation_params(sim_name, i):
    import quaternion
    quat = quaternion.as_quat_array(cached_get_quat(sim_name)[i])
    omega_mb = cached_get_omega_mb(sim_name)[i]
    pivot = get_pivot(sim_name)
    return quat, omega_mb, pivot

@st.cache(show_spinner=False, suppress_st_warning=True)
def get_peri_idx(tbl):
    r = np.linalg.norm(np.vstack([tbl['x'],tbl['y']]).T, axis=1)
    # print(r)
    # peri = r.argmin()
    zero_crossings = np.where(np.diff(np.signbit(np.diff(r))))[0]
    try:
        # idx_peri = zero_crossings[0]  # in pandas first element is always 0,here we use np and the first zero_crossing is actually the pericenter
        idx_peri1 = zero_crossings[0]  # pericenter
        idx_peri2 = zero_crossings[2]

    except IndexError as e:
        st.warning('There seems no pericenter yet: (' + str(e) +')')
        idx_peri1 = idx_peri2 = 0

    # idx_peri1 = zero_crossings[0]  # pericenter
    # idx_peri2 = zero_crossings[2]
    # idx_peri3 = zero_crossings[3]  # first element is always 0, so the second is the actual pericenter
    return idx_peri1, idx_peri2


def plot_traj(x, y, i, ax, peri_idx):

    # x[i], y[i]
    ax.plot(x[:i], y[:i], 'g')
    ax.plot(x[i:], y[i:], '--r')


    ax.plot((0,0),'b+')
#     ax.plot(*traj[0, :], 'go', markersize=5)
#     ax.plot(*traj[-2, :],'ro', markersize=5)
    # ax.set_xlabel('x [kpc]')
    # ax.set_ylabel('y [kpc]')
    # print(np.vstack([x,y]))
    r = np.linalg.norm(np.vstack([x,y]).T, axis=1)

    ax.scatter(x[peri_idx[0]], y[peri_idx[0]], color="k", marker='.')
    ax.scatter(x[peri_idx[1]], y[peri_idx[1]], color="k", marker='.')

    ax.set_aspect('equal')
    lim = 1000
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.scatter(x[i], y[i], color="red", marker='o', facecolors='none')


def plot_rotated_axis(rm, ax):
    head_width = 6e-2
    dashed = rm[2][:] < 0
    # print(dashed)
    ls = list()
    for d in dashed:
        ls.append(':' if d else '-')
    ax.arrow(0,0, rm[0, 0], rm[1, 0], length_includes_head=True, head_width=head_width, color='r', ls=ls[0])
    ax.arrow(0,0, rm[0, 1], rm[1, 1], length_includes_head=True, head_width=head_width, color='g', ls=ls[1])
    ax.arrow(0,0, rm[0, 2], rm[1, 2], length_includes_head=True, head_width=head_width, color='b', ls=ls[2])
    ax.set_aspect('equal')
    lim=1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ytext = 0.65
    xtext = 0.3
    ax.text(xtext,     ytext, 'X', c='r', size='x-small')
    ax.text(xtext+0.2, ytext, 'Y', c='g', size='x-small')
    ax.text(xtext+0.4, ytext, 'Z', c='b', size='x-small')

def add_cluster_center_velocity_directions(ax_sb, orbital_position_rotated, orbital_velocity_rotated, color_cluster='w', color_velocity='g', with_text=True):
    cl_direction = -normify(orbital_position_rotated[0:2], 8)
    v_orbit_direction = normify(orbital_velocity_rotated[0:2], 8)
    # st.write(cl_direction)
    _arrow_factor = 1.4
    center = np.array([0.5, 0.5])
    ax_sb.arrow(*(center+cl_direction*_arrow_factor), *cl_direction,
        width=4e-3,
        # head_width=1e-2,
        length_includes_head=False, color=color_cluster, transform=ax_sb.transAxes)
    ax_sb.arrow(*(center+v_orbit_direction*_arrow_factor), *v_orbit_direction,
        width=4e-3,
        # head_width=1e-2,
        length_includes_head=False, color=color_velocity, transform=ax_sb.transAxes, ls='--')
    if with_text:
        xtext, ytext = 0, 0.97
        ax_sb.text(xtext, ytext, 'Velocity', c=color_velocity, size='x-small', weight='semibold', transform=ax_sb.transAxes)
        ax_sb.text(xtext, ytext-0.03, 'Cluster center', c=color_cluster, size='x-small', weight='semibold', transform=ax_sb.transAxes)

@st.cache(show_spinner=False)
def get_trajectory(tbl, i):
    x, y, z = tbl['x'], tbl['y'], tbl['z']
    cur_pos = x[i], y[i], z[i]
    cur_vel = tbl['vx'][i], tbl['vy'][i], tbl['vz'][i]
    r = np.linalg.norm(np.array(cur_pos))
    v = np.linalg.norm(np.array(cur_vel))
    return x, y, z, cur_pos, cur_vel, r, v

@st.cache(show_spinner=False)
def get_pos_vel(tbl, i):
    x, y, z = tbl['x'], tbl['y'], tbl['z']
    cur_pos = x[i], y[i], z[i]
    cur_vel = tbl['vx'][i], tbl['vy'][i], tbl['vz'][i]
    orbital_position = np.array(cur_pos)
    orbital_velocity = np.array(cur_vel)
    return orbital_position, orbital_velocity

def pos_vel(tbl_traj, i):
    t = tbl_traj[i]
    return np.array([t['x'], t['y'], t['z']]), np.array([t['vx'],t['vy'], t['vz']])


def do_derotate(snap, sim_name, which_snap):
    quat_mb, omega_mb, pivot = get_correct_derotation_params(sim_name, which_snap-1)
    # Initial rotation is already included. The following is not needed
    # quat_vp0 = get_initial_rotation(sim_name)
    # quat = quat_mb * quat_vp0.inverse()
    quat = quat_mb
    # st.info(quat)
    # note that np.quaternion is in scalar-first notation, while `scipy.spatial.transform.Rotation.from_quat` is
    # in scalar-last notation. So I can't use `quaternion.as_float_array`
    # (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_quat.html)
    quat_arr = np.array([quat.x, quat.y, quat.z, quat.w])
    # print(quat_arr)
    quat_rot = R.from_quat(quat_arr)
    snap['pos'], snap['vel'] = derotate_pos_and_vel(snap['pos'], snap['vel'], quat, omega_mb, pivot)

    # snap = rotate_snap(snap, quat, omega_mb, pivot, on_orbit_plane=False)
    return snap, quat_rot

def get_snap(which_snap, sim_name, derotate):
    if platform.node() == 'gandalf':
        dirname = "/home/mmastrop/SIM"
    else:
        dirname = "/home/michele/sim/MySimulations/ok_new_adhoc_or_not_affected"

    snap_name = os.path.join(dirname, sim_name, f'out/snapshot_{which_snap:04}')

    if pynbody.__version__ == '0.46':
        snap_orig = pynbody.load(snap_name)
    else:
        snap_orig = pynbody.load(snap_name, ignore_cosmo=True)

    if derotate:
        snap, quat_rot = do_derotate(snap_orig, sim_name, which_snap)
    else:
        snap, quat_rot = snap_orig, None

    return snap, quat_rot



def get_sb(snap, sb_range, band, width, resolution, show_cbar=True, lum_pc2=False):
    mag_filter = sb_range[1]
    bad_pixels = 'black'
    return surface_brightness(snap, band=band, lum_pc2=lum_pc2, width=width, resolution=resolution,
                            mag_filter=mag_filter, cmap_name='nipy_spectral', noplot=True,
                            vmin=sb_range[0], vmax=sb_range[1],
                            bad_pixels=bad_pixels, show_cbar=show_cbar)


def create_fig_sb(snap, ax_sb, sb_range, band, width, resolution, show_cbar=True):
    # mag_filter = st.sidebar.slider('Magnitude filter (do not plot fainter than ... mag/arcsec^2)',
    #                                 min_value=20, max_value=sb_range[1], value=sb_range[1])
    mag_filter = sb_range[1]
    # bad_pixels = st.sidebar.selectbox('Bad Pixels', options=['black', 'white'])
    bad_pixels = 'black'
    sb = surface_brightness(snap, band=band, subplot=ax_sb, width=width, resolution=resolution,
                            mag_filter=mag_filter, cmap_name='nipy_spectral',
                            vmin=sb_range[0], vmax=sb_range[1],
                            bad_pixels=bad_pixels, show_cbar=show_cbar)
    ax_sb.grid(ls=':')

    return sb



def first_inset(ax_sb, x,y, which_snap, peri_idx):
    axins = inset_axes(ax_sb, **INSET_DIM)
    plot_traj(x,y, which_snap-1, axins, peri_idx)

def axis_inset(ax_sb, rm, vel):
    ax_arrow = inset_axes(ax_sb, **INSET_DIM, loc='lower right')
    # st.write(rm)
    plot_rotated_axis(rm, ax_arrow)
    head_width = 6e-2

    v = normify(vel,2)
    ax_arrow.arrow(0,0, v[0], v[1], length_includes_head=True, head_width=head_width, color='k', ls=':' if vel[2] < 0 else '-')
    ytext = -0.9
    xtext = 0.3
    ax_arrow.text(xtext+0.4, ytext, 'V', c='k', size='x-small')

def rotated_traj_inset(ax_sb, which_snap, x, y, rotation_and_moving_box, peri_idx):
    traj_3d = np.vstack([x, y, np.zeros_like(x)]).T
    rot_traj_3d = rotation_and_moving_box.apply(traj_3d)
    # st.write(traj_3d[which_snap])
    # st.write(cur_pos)
    axins2 = inset_axes(ax_sb, **INSET_DIM, loc='right')
    plot_traj(rot_traj_3d[:,0], rot_traj_3d[:,1], which_snap-1, axins2, peri_idx)

def create_stellar_velocity_map(snap, sb, ax_star_vel, star_v_range, width, resolution,
                                minimum_sb=None, astronomical_convention=False):
    sign = -1 if astronomical_convention else 1
    star_vel = sign * pynbody.plot.sph.image(snap.s, qty='vz', av_z='v_lum_den',
                                  width=width, resolution=resolution, noplot=True)
    extent = (-width/2, width/2, -width/2, width/2)

    star_vel[np.isnan(sb)] = np.nan
    if minimum_sb is not None:
        star_vel[sb > minimum_sb] = np.nan

    _img_v_star = ax_star_vel.imshow(star_vel-np.nanmean(star_vel), cmap='sauron', extent=extent, origin='lower',
                                norm=Normalize(*star_v_range))

    cbar = ax_star_vel.figure.colorbar(_img_v_star)
    cbar.set_label(r'$v_z^{\star}$ (km/s)')

    ax_star_vel.set_xlabel('x/kpc')
    ax_star_vel.set_ylabel('y/kpc')
    ax_star_vel.set_title(r'Stellar LOS')
    ax_star_vel.grid(ls=':')


def cii_map(snap, width, resolution):
    cii = pynbody.plot.sph.image(snap.g, qty='cii', units='erg s**-1 cm**-2',
                                     width=width, resolution=resolution, noplot=True)
    return cii

def create_cii_map(snap, ax, width, resolution, cii_range):
    cii = cii_map(snap, width, resolution)
    extent = (-width/2, width/2, -width/2, width/2)
    cii[cii<cii_range[0]] = np.nan

    _img = ax.imshow(cii, cmap='jet', extent=extent, origin='lower',
                     norm=LogNorm(*cii_range))

    cbar = ax.figure.colorbar(_img)
    cbar.set_label(r'[CII] (erg s$^{-1}$ cm$^{-2})$')

    ax.set_xlabel('x/kpc')
    ax.set_ylabel('y/kpc')
    ax.set_title(r'[CII]')
    ax.grid(ls=':')


def age_map(snap, sb, width, resolution, av_z='v_lum_den'):
    age = pynbody.plot.sph.image(snap.s, qty='age', av_z=av_z, units='Gyr',
                                  width=width, resolution=resolution, noplot=True)
    age[np.isnan(sb)] = np.nan
    return age

def create_age_map(snap, sb, ax, width, resolution, av_z='v_lum_den'):
    age = age_map(snap, sb, width, resolution, av_z=av_z)
    extent = (-width/2, width/2, -width/2, width/2)
    _img = ax.imshow(age, cmap='jet', extent=extent, origin='lower',
                                norm=Normalize(0, None)
                                )

    cbar = ax.figure.colorbar(_img)
    cbar.set_label(r'age (Gyr)')

    ax.set_xlabel('x/kpc')
    ax.set_ylabel('y/kpc')
    ax.set_title(r'Age')
    ax.grid(ls=':')

def get_beam_patch(ellipse_smooth, width, resolution):
    stddev = np.array(ellipse_smooth) * 0.1  # arcsec -> kpc
    cen = np.array([-width/2, -width/2]) + stddev/2 #+ 5*(resolution / width)

    # print(cen)
    beam = Ellipse(cen, *stddev, facecolor='none', edgecolor='r')
    return beam



def get_hi(snap, width, resolution, ellipse_smooth=None):
    """
    ellipse_smooth:tuple
        in arcsec

    """
    hi = pynbody.plot.sph.image(snap.g,
                                qty='rho_HI', units='amu cm**-2',
                                width=width, resolution=resolution, noplot=True)
    if ellipse_smooth is not None:
        stddev = np.array(ellipse_smooth) * 0.1  # arcsec -> kpc
        # print(stddev)
        pixels = resolution * stddev / width
        # print(pixels)
        gauss_kernel = Gaussian2DKernel(*pixels/2)
        # print('smoothing', pixels)
        hi_img = convolve(hi, gauss_kernel)
    else:
        hi_img = hi
    return hi_img


def create_hi_image(snap, ax_hi, width, resolution, ellipse_smooth=None, vmin=1e19, vmax=1e22, ret_im=False):
# hi = pynbody.plot.sph.image(snap.g, qty='rho_HI', units='amu cm**-2',
#                             width=width, resolution=resolution,
#                             subplot=ax_hi, cmap='gray', vmin=1e19, vmax=1e22)

    if ret_im:
        return pynbody.plot.sph.image(snap.g,
                                      qty='rho_HI', units='amu cm**-2',
                                      width=width, resolution=resolution, noplot=True)

    extent = (-width/2, width/2, -width/2, width/2)
    ax_hi.set_title(r'$\Sigma_{HI}$')
    if ellipse_smooth is not None:
        hi = pynbody.plot.sph.image(snap.g,
                                    qty='rho_HI', units='amu cm**-2',
                                    width=width, resolution=resolution, noplot=True)
        # print(hi)
        stddev = np.array(ellipse_smooth) * 0.1  # arcsec -> kpc
        # print(stddev)
        pixels = resolution * stddev / width
        # print(pixels)
        gauss_kernel = Gaussian2DKernel(*pixels/2)
        # print('smoothing', pixels)
        hi_img = convolve(hi, gauss_kernel)
        _img = ax_hi.imshow(hi_img, cmap='gray', extent=extent, origin='lower',
                            norm=LogNorm(vmin=vmin, vmax=vmax))
        cbar = ax_hi.figure.colorbar(_img)
        cbar.set_label(r'$\Sigma_{HI}$ (amu/cm$^2$)')

        ax_hi.set_xlabel('x/kpc')
        ax_hi.set_ylabel('y/kpc')

    else:
        hi_img = pynbody.plot.sph.image(snap.g, qty='rho_HI', units='amu cm**-2',
                                        width=width, resolution=resolution,
                                        subplot=ax_hi,
                                        cmap='gray',
                                        vmin=vmin, vmax=vmax,
                                        )
    ax_hi.grid(ls=':')

    return hi_img

def create_hi_vel_image(snap, ax_velocity, hi_img, hi_v_range, min_sigma_hi_for_velocity_plot, rm, width, resolution, astronomical_convention=False):
    sign = -1 if astronomical_convention else 1

    hi_v = sign * pynbody.plot.sph.image(snap.g, qty='vz', av_z='rho_HI',
                                  width=width, resolution=resolution, noplot=True)
    extent = (-width/2, width/2, -width/2, width/2)

    # min_sigma_hi_for_velocity_plot = levels_hi.min()
    hi_v[hi_img < min_sigma_hi_for_velocity_plot] = np.nan
    _img_v = ax_velocity.imshow(hi_v-np.nanmean(hi_v), cmap='jet', extent=extent, origin='lower',
                                norm=Normalize(*hi_v_range))

    cbar = ax_velocity.figure.colorbar(_img_v)
    cbar.set_label(r'$v_z^{HI}$ (km/s)')

    ax_velocity.set_xlabel('x/kpc')
    ax_velocity.set_ylabel('y/kpc')
    ax_velocity.set_title(r'HI velocity')
    ax_velocity.grid(ls=':')

    ax_arrow2 = inset_axes(ax_velocity, **INSET_DIM, loc='lower right')
    plot_rotated_axis(rm, ax_arrow2)


def do_sb_contours(sb, levels_star, ax, width, colors='r'):
    extent = (-width/2, width/2, -width/2, width/2)
    mysb = sb.copy()
    mysb[np.isnan(mysb)] = np.nanmin(sb)
    cs_sb = ax.contour(mysb, extent=extent,
                       levels=levels_star, colors=colors,
                       linewidths=0.8, linestyles='--', alpha=0.6)
    return cs_sb

def do_hi_contours(hi_img, levels_hi, ax, width, colors='w'):
    extent = (-width/2, width/2, -width/2, width/2)
    cs_hi = ax.contour(hi_img, extent=extent,
                       levels=levels_hi, colors=colors,
                       linewidths=0.8, linestyles='--', alpha=0.8)
    return cs_hi

def do_contours(sb, hi_img, levels_hi, levels_star, ax_sb, ax_color, ax_star_vel, ax_hi, ax_velocity, width):
    extent = (-width/2, width/2, -width/2, width/2)

    # contours
    contour_axis = [ax_sb, ax_color, ax_star_vel] if ax_color is not None else [ax_sb, ax_star_vel]
    contour_colors = ['w', 'k', 'k'] if ax_color is not None else ['w', 'k']
    # contour_axis = [ax_sb, ax_color, ax_star_vel]
    # contour_colors = ['w', 'k', 'k']

    for col, _ax in zip(contour_colors, contour_axis):
        if _ax is None: continue
        cs_hi = _ax.contour(hi_img, extent=extent,
                         levels=levels_hi, colors=col,
                         linewidths=0.8, linestyles='--', alpha=0.6)

    # remove nan
    mysb = sb.copy()
    mysb[np.isnan(mysb)] = np.nanmin(sb)
    for _ax1 in ax_hi, ax_velocity:
        cs_sb = _ax1.contour(mysb, extent=extent,
                             levels=levels_star, colors='r',
                             linewidths=0.8, linestyles='--', alpha=0.6)


def get_ngc1427A_angular_distance():
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    # ngc1427a = SkyCoord.from_name('ngc1427a')
    ngc1427a = SkyCoord(55.0375*u.deg, -35.62611*u.deg, frame='icrs')
    # ngc1399 = SkyCoord.from_name('ngc1399')
    ngc1399 = SkyCoord(54.6211792*u.deg, -35.4507417*u.deg, frame='icrs')

    sep = ngc1427a.separation(ngc1399)
    # @20 Mpc 1'' = 0.1 kpc
    projected_dist_arcsec = sep.to('arcsec')
    projected_dist_kpc = projected_dist_arcsec * 0.1*u.kpc/u.arcsec
    # print(projected_dist_arcsec, projected_dist_kpc)
    # 137 kpc
    return projected_dist_kpc, projected_dist_arcsec


def phitheta2rotation(phi, theta, orbital_position=None):
    """The input phi and theta should be in deg"""
    r1 = R.from_rotvec(phi*np.pi/180 * np.array([0, 0, 1]))
    y_prime = r1.apply(np.array([0, 1, 0]))
    r2 = R.from_rotvec(theta*np.pi/180 * y_prime)

    axis_rotation = (r2 * r1).inv()
    if orbital_position is not None:
        orbital_position_rotated_plane = axis_rotation.apply(orbital_position)
        z_rot = np.arctan2(orbital_position_rotated_plane[1], orbital_position_rotated_plane[0])
        # Add an offset of 210 degree to fix the cluster center direction to north-east
        _value_z_rot = float((-z_rot*180.0/np.pi+210.0) % 360.0)
        oz = st.sidebar.slider('Rotate z (default of cluster center inclined of 30deg)', min_value=0.0, max_value=360.0, value=_value_z_rot, step=2.0)
        r3 = R.from_euler('z', oz, degrees=True)
        rotation = r3 * axis_rotation
    else:
        rotation = axis_rotation
    return rotation

def get_solved_rotation(orbital_position, orbital_velocity, rp, vp, which_solution, sign_of_r):
    w = which_solution
    phi_tuple, theta_tuple = solve_the_problem(
        r_arr=orbital_position,
        v_arr=orbital_velocity,
        rp=rp,
        vp=vp,
        sign_of_r=sign_of_r)
    phi = phi_tuple[w-1]
    theta = theta_tuple[w-1]
    # st.write(r"$\varphi$:", phi, r"$\theta$:",  theta))
    # st.write(sol)

    if np.isnan(theta):
        st.error('No solution for this snap')
        return None, phi, theta
    else:
        r1 = R.from_rotvec(phi*np.pi/180 * np.array([0, 0, 1]))

        y_prime = r1.apply(np.array([0, 1, 0]))
        r2 = R.from_rotvec(theta*np.pi/180 * y_prime)

        axis_rotation = (r2 * r1).inv()
        orbital_position_rotated_plane = axis_rotation.apply(orbital_position)
        orbital_velocity_rotated_plane = axis_rotation.apply(orbital_velocity)
        z_rot = np.arctan2(orbital_position_rotated_plane[1], orbital_position_rotated_plane[0])
        # Add an offset of 210 degree to fix the cluster center direction to north-west
        _value_z_rot = float((-z_rot*180.0/np.pi+210.0) % 360.0)
        oz = st.sidebar.slider('Rotate z (default of cluster center inclined of 30deg)', min_value=0.0, max_value=360.0, value=_value_z_rot, step=2.0)
        r3 = R.from_euler('z', oz, degrees=True)
        rotation = r3 * axis_rotation
    return rotation, phi, theta




def solve_problem_for_trajectory(tbl_traj, rp, vp, sign_of_r):
    from collections import defaultdict
    import pandas as pd

    d = defaultdict(list)

    for t in tbl_traj:
        r_arr = np.array([t['x'], t['y'], t['z']])
        v_arr = np.array([t['vx'], t['vy'], t['vz']])

        # try:
        p = Problem4(r=r_arr, v=v_arr, rp=rp, vp=vp, sign=sign_of_r)
        x, y, z = p.x, p.y, p.z
        # except RuntimeError:
        #     x = y = z = np.array([np.nan, np.nan])

        d['x1'].append(x[0])
        d['y1'].append(y[0])
        d['z1'].append(z[0])
        d['x2'].append(x[1])
        d['y2'].append(y[1])
        d['z2'].append(z[1])

    df = pd.DataFrame(d)

    # import pdb;pdb.set_trace()
    # df = pd.concat([tbl_traj.to_pandas(), pd.DataFrame(d)], axis=1)

    df['phi1'] = np.arctan2(df.y1, df.x1)*180/np.pi
    df['phi2'] = np.arctan2(df.y2, df.x2)*180/np.pi
    df['theta1'] = np.arccos(df.z1)*180/np.pi
    df['theta2'] = np.arccos(df.z2)*180/np.pi
    return df


def solve_problem_for_trajectory_no_sign(tbl_traj, rp, vp, sign_of_r=None):
    from collections import defaultdict
    import pandas as pd

    d = defaultdict(list)

    for t in tbl_traj:
        r_arr = np.array([t['x'], t['y'], t['z']])
        v_arr = np.array([t['vx'], t['vy'], t['vz']])

        # try:
        pm = Problem4(r=r_arr, v=v_arr, rp=rp, vp=vp, sign=-1)
        pp = Problem4(r=r_arr, v=v_arr, rp=rp, vp=vp, sign=1)

        d['xm'].append(pm.x)
        d['ym'].append(pm.y)
        d['zm'].append(pm.z)
        d['xp'].append(pp.x)
        d['yp'].append(pp.y)
        d['zp'].append(pp.z)

    df = pd.DataFrame(d)
    # import pdb;pdb.set_trace()

    are_both_signs_valid = np.logical_and(np.logical_not(np.isnan(np.vstack(df['xm']))),
                                          np.logical_not(np.isnan(np.vstack(df['xp']))))
    if not (are_both_signs_valid[:,0] == are_both_signs_valid[:,1]).all():
        raise RuntimeError('Problem in are_both_valid')
    else:
        are_both_valid = are_both_signs_valid[:,0]

    st.write(are_both_valid)
    if are_both_valid.any():
        st.write(df['xm'][are_both_valid])
        st.write(df['xp'][are_both_valid])
        raise RuntimeError('I have four solutions, not two')
    # import pdb;pdb.set_trace()
    # df = pd.concat([tbl_traj.to_pandas(), pd.DataFrame(d)], axis=1)

    df['phi1'] = np.arctan2(df.y1, df.x1)*180/np.pi
    df['phi2'] = np.arctan2(df.y2, df.x2)*180/np.pi
    df['theta1'] = np.arccos(df.z1)*180/np.pi
    df['theta2'] = np.arccos(df.z2)*180/np.pi
    return df


def solve_the_problem(r_arr, v_arr, rp, vp, sign_of_r):
    p = Problem4(r=r_arr, v=v_arr, rp=rp, vp=vp, sign=sign_of_r)
    # print(p)
    # st.write(p, p.num, p.a, p.b)
    phi = np.array([np.arctan2(p.y[0], p.x[0]), np.arctan2(p.y[1], p.x[1])])
    theta = np.array([np.arccos(p.z[0]), np.arccos(p.z[1])])
    return phi*180/np.pi, theta*180/np.pi


REAL_ALPHA = 45
REAL_BETA = 120
def filter_out(df, tol, real_alpha=REAL_ALPHA, real_beta=REAL_BETA):
    return df.query(f'{real_alpha-tol}<alpha<{real_alpha+tol} & {real_beta-tol}<beta<{real_beta+tol}')


def get_data_name(cache_file):
    _dir_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(_dir_path, 'data', cache_file)


def get_data(cache_file):
    return pd.read_pickle(get_data_name(cache_file))


def savefig(fig, file_stem, ext, dpi=300, **kwargs):
    file_name = file_stem + ext
    print(f'Saving {file_name}...')
    if ext == '.png':
        fig.savefig(file_name, dpi=dpi, bbox_inches='tight', **kwargs)
        out = f"{file_stem}-crop.png"
        os.system(f'convert -trim {file_name} {out}')
    elif ext == '.pdf':
        fig.savefig(file_name, dpi=dpi, bbox_inches='tight', **kwargs)
        os.system(f'pdfcrop {file_name}')
