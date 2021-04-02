import numpy as np

def get_aperture_from_moments(img, width, resolution, ax=None):
    from photutils import data_properties, EllipticalAperture
    import astropy.units as u
    # https://photutils.readthedocs.io/en/stable/api/photutils.segmentation.SourceProperties.html
    cat = data_properties(img)
    columns = ['id', 'xcentroid', 'ycentroid', 'semimajor_axis_sigma',
               'semiminor_axis_sigma', 'orientation']
    tbl = cat.to_table(columns=columns)
    tbl['xcentroid'].info.format = '.10f'  # optional format
    tbl['ycentroid'].info.format = '.10f'
    tbl['semiminor_axis_sigma'].info.format = '.10f'
    tbl['orientation'].info.format = '.10f'
    # print(tbl)

    position = (np.array((cat.xcentroid.value, cat.ycentroid.value)) - resolution/2) * width/resolution
    # semiminor axis: The 1-sigma standard deviation along the semimajor axis of the 2D Gaussian function that has the same second-order central moments as the source.

    a = cat.semimajor_axis_sigma.value * width/resolution
    b = cat.semiminor_axis_sigma.value * width/resolution
    theta = cat.orientation.to(u.rad).value
    apertures = EllipticalAperture(position, a, b, theta=theta)
    # print(apertures)
    if ax is not None:
        apertures.plot(color='#d62728', axes=ax)
    # return a, b, theta
    return apertures


if __name__ == '__main__':
    import copy
    import matplotlib.pyplot as plt
    from simulation.simdata import get_traj
    from ngc1427apaper.helper import get_snap
    from interface import Interface, get_rotation
    DPI = 300
    ii = Interface()
    derotate = True
    snap, quat_rot = get_snap(ii.which_snap, ii.sim_name, derotate)
    tbl_traj = get_traj(ii.sim_name)
    orbital_position, orbital_velocity = get_pos_vel(tbl_traj, ii.which_snap-1)
    if ii.do_rotate_to_solution:
        rotation, phi, theta = get_solved_rotation(orbital_position, orbital_velocity, ii.rp, ii.vp, ii.w, ii.sign_of_r)
        if rotation is None:
            rotation = get_rotation()
    else:
        rotation = get_rotation()

    rotation_matrix = rotation.as_matrix()
    orbital_position_rotated = rotation.apply(orbital_position)
    orbital_velocity_rotated = rotation.apply(orbital_velocity)

    pynbody.analysis.halo.center(snap.s, vel=False)
    tr = pynbody.transformation.GenericRotation(snap, rotation_matrix)

    fig_hi, ax_hi = plt.subplots(dpi=DPI)
    hi_img = create_hi_image(snap, ax_hi, ii.width, ii.resolution, ellipse_smooth=ii.ellipse_smooth)

    if ii.show_beam:
        beam = get_beam_patch(ii.ellipse_smooth, ii.width, ii.resolution)
        ax_hi.add_patch(copy.copy(beam))
        # ax_hi_velocity.add_patch(copy.copy(beam))


    # a, b, theta = get_apertures_from_moments(hi_img, width, resolution, ax_hi)
    aperture = get_aperture_from_moments(hi_img, ii.width, ii.resolution, ax_hi)
    st.write(aperture)
    st.write(fig_hi)
