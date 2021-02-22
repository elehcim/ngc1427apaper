#! /usr/bin/env python
import copy
import matplotlib.pyplot as plt
from simulation.ellipse_fit import fit_contour, ellipseparams2patch
from simulation.simdata import get_traj
from simulation.simdata import SIM_NAME_DICT, SPECIAL_DICT

from ngc1427apaper.helper import *
from ngc1427apaper.hi_tail import get_aperture_from_moments

from ngc1427apaper.angles import ellipseparams2aperture, get_angle, draw_arrows_from_vec, add_corner_arc
from ngc1427apaper.globals import *

SIM_DICT = {**SIM_NAME_DICT, **SPECIAL_DICT}


def filter_out_with_velocity(df, tol, iso=1, real_alpha=REAL_ALPHA, real_beta=REAL_BETA):
    rx, ry = np.cos(np.pi/3), np.sin(np.pi/3)

    l = df.query(f'{real_alpha-tol}<alpha{iso}<{real_alpha+tol} & {real_beta-tol}<beta{iso}<{real_beta+tol} &\
                   (vx_rot * {rx} + vy_rot * {ry})>0 & not sim.str.endswith("r") & hi_max > 1e15\
                   ').copy()
                   # & nearest_peri == 2\
    l.loc[:,'tol'] = tol
    l.loc[:,'iso'] = isophote_target[iso]
    # print(l)
    return l


if __name__ == '__main__':
    dff = get_data('cache_with_iso_and_hi_valid.pkl')
    df = filter_out_with_velocity(dff, tol=15)

    row = df.query('sim == "68p100" & snap == 49 & rp == 77 & vp == -793 & sol == 2 & sign == 1')
    print(row)
    if len(row) == 1:
        ii = row.iloc[0]
    else:
        raise RuntimeError("Can't find unique oriented snapshot")
    sim_label = ii['sim']
    sim_name = SIM_DICT[sim_label]
    which_snap = ii['snap']

    width = 35
    resolution = 500

    derotate = True
    snap, quat_rot = get_snap(which_snap, sim_name, derotate)
    tbl_traj = get_traj(sim_name)
    orbital_position, orbital_velocity = get_pos_vel(tbl_traj, which_snap-1)
    rotation = phitheta2rotation(ii.phi, ii.theta, orbital_position=orbital_position)


    rotation_matrix = rotation.as_matrix()
    orbital_position_rotated = rotation.apply(orbital_position)
    orbital_velocity_rotated = rotation.apply(orbital_velocity)

    pynbody.analysis.halo.center(snap.s, vel=False)
    tr = pynbody.transformation.GenericRotation(snap, rotation_matrix)

    fig_sb, ax_sb = plt.subplots()
    sb_range = (21.0, 27.5)
    band = 'sdss_r'
    sb = create_fig_sb(snap, ax_sb, sb_range, band, width, resolution)
    # ax_sb.set_title(rf'$\mu_{{{band}}}$ - ({which_snap}) - {time:.2f} Gyr')
    print(f"{sim_label}s{which_snap}")
    # ax_sb.set_title(rf"$\mu_{{{band}}}$ - {sim_label}s{which_snap:03} - $\tau$={tbl_traj['t_period'][which_snap-1]:.4f}")

    isophote_sb = 27
    ell = fit_contour(sb, isophote_sb, width, resolution)
    ep = ellipseparams2patch(ell, edgecolor='blue', linewidth=2)
    aperture_sb = ellipseparams2aperture(ell)

    ellipse_smooth = None#(10, 10)
    ax_sb.add_patch(ep)
    fig_hi, ax_hi = plt.subplots()
    hi_range = (1e18, None)
    hi_img = create_hi_image(snap, ax_hi, width, resolution, vmin=hi_range[0], vmax=hi_range[1], ellipse_smooth=ellipse_smooth)
    # ax_hi.set_title(r'$\Sigma_{HI}$')

    # if ii.show_beam:
    if ellipse_smooth is not None:
        beam = get_beam_patch(ellipse_smooth, width, resolution)
        ax_sb.add_patch(copy.copy(beam))

    levels_hi = np.logspace(17, 21, 5)

    for c, ax in zip(('w', 'k'), (ax_sb,)):
        do_hi_contours(hi_img, levels_hi, ax, width, colors=c)

    aperture_hi = get_aperture_from_moments(hi_img, width, resolution)

    fac = 3
    theta_pos = ii['theta_pos']
    # theta_pos
    theta_pos *= np.pi/180
    # p = -normify(np.array([np.cos(theta_pos), np.sin(theta_pos)]), fac=2)

    p = -normify(orbital_position_rotated[0:2], fac)
    v = normify(orbital_velocity_rotated[0:2], fac)
    h = normify(np.array([np.cos(aperture_hi.theta), np.sin(aperture_hi.theta)]), fac=fac)
    s = normify(np.array([np.cos(aperture_sb.theta), np.sin(aperture_sb.theta)]), fac=fac)
    alpha = get_angle(p, s)
    beta = get_angle(h, s)

    # st.write(p, v, h, s)
    print(rf'$\alpha$ = {alpha:.2f}, $\beta$ = {beta:.2f}')

    # zero = np.array((0, 0))

    # add_corner_arc(ax_sb, [p, zero, s],
    #                radius=.5, color='r', text=int(alpha),
    #                text_radius=0.3, text_color='w',
    #                alpha=0.8, ls='-.')
    # add_corner_arc(ax_sb, [h, zero, s],
    #                radius=.4, color='w', text=int(beta),
    #                text_radius=0.3, text_color='w',
    #                alpha=0.8, ls='-.')
    # draw_arrows(ax_sb, orbital_position_rotated, orbital_velocity_rotated, aperture_hi.theta, aperture_sb.theta)
    draw_arrows_from_vec(ax_sb, p, v, -h, -s, color_p='w')

    # ax_hi
    # hi_img = create_hi_image(snap, ax_hi, width, resolution, vmin=None, vmax=None, ellipse_smooth=ellipse_smooth)

    fig_hi_vel, ax_hi_vel = plt.subplots()

    min_sigma_hi_for_velocity_plot = 5e+18
    hi_v_range = (None, None)
    create_hi_vel_image(snap,
        ax_hi_vel,
        hi_img,
        hi_v_range,
        min_sigma_hi_for_velocity_plot,
        width,
        resolution,
        astronomical_convention=True)

    levels_star = np.linspace(22, 28, 4)
    do_sb_contours(sb, levels_star, ax_hi_vel, width)

    savefig(fig_sb, "SB", '.png', dpi=300)
    savefig(fig_sb, "SB", '.pdf', dpi=300)
    savefig(fig_hi, "HI", '.png', dpi=300)
    savefig(fig_hi, "HI", '.pdf', dpi=300)
    savefig(fig_hi_vel, "HI_VEL", '.png', dpi=300)
    savefig(fig_hi_vel, "HI_VEL", '.pdf', dpi=300)
    # plt.show()
    # st.write(fig_hi)
