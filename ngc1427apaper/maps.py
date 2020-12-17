import os
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import pynbody
import streamlit as st
from scipy.spatial.transform import Rotation as R
from simulation.simdata import SIM_NAME_DICT, SPECIAL_DICT, get_traj

from ngc1427apaper.helper import get_snap, get_trajectory, get_peri_idx, create_fig_sb, create_hi_image, \
    create_stellar_velocity_map, add_cluster_center_velocity_directions, do_hi_contours, create_hi_vel_image

DPI = 300

levels_hi = np.logspace(17, 22, 5)
levels_star = np.linspace(22, 28, 4)

SIM_DICT = {**SIM_NAME_DICT, **SPECIAL_DICT}


class PreparedSnap:
    def __init__(self, sim_label, which_snap, rot=(145, 50, -95), band='sdss_r',
                 derotate=True, width=50, resolution=400):
        self.sim_label = sim_label
        self.which_snap = which_snap
        sim_name = SIM_DICT[sim_label]
        self.sim_name = sim_name
        self.prefix = f'{sim_label}_{which_snap:04}'
        self.band = band

        self.snap, quat_rot = get_snap(which_snap, sim_name, derotate)

        self.width = width #st.sidebar.slider('Width (kpc)', min_value=1, max_value=80, value=50, step=5)
        self.resolution = resolution
        # st.sidebar.slider('Resolution', min_value=100, max_value=500, value=200, step=100)
        self._rot = rot
        ox, oy, oz = rot
        rotation = R.from_euler('ZYX', [oz, oy, ox], degrees=True)
        self.rm = rotation.as_matrix()
        if quat_rot is not None:
            rotation_and_moving_box = rotation * quat_rot.inv()
        else:
            rotation_and_moving_box = rotation
        self.full_rotation_matrix = rotation_and_moving_box.as_matrix()
        # print(full_rotation_matrix)

        pynbody.analysis.halo.center(self.snap.s, vel=False)

        tr = pynbody.transformation.GenericRotation(self.snap, self.full_rotation_matrix)
        # tr.apply()

        tbl_traj = get_traj(sim_name)
        self.tbl_traj = tbl_traj

        self.x, self.y, self.z, self.cur_pos, self.cur_vel, self.r, self.v = get_trajectory(tbl_traj, which_snap)
        self.peri_idx = get_peri_idx(tbl_traj)
        orbital_position = np.array([*self.cur_pos])
        orbital_velocity = np.array([*self.cur_vel])
        # print(orbital_position)
        self.orbital_position_rotated = rotation_and_moving_box.apply(orbital_position)
        self.orbital_velocity_rotated = rotation_and_moving_box.apply(orbital_velocity)
        self.r_xy = np.linalg.norm(self.orbital_position_rotated[0:2])
        self.los_velocity = -self.orbital_velocity_rotated[2]  # minus for astronomical convention

        self.time = tbl_traj["t"][which_snap]
        self.time_since_peri = self.time - tbl_traj["t"][self.peri_idx[0]]
        self.time_since_2nd_peri = self.time - tbl_traj["t"][self.peri_idx[1]]
        self.tau = self.tbl_traj['t_period'][which_snap]


class SolvedPreparedSnap:
    def __init__(self, sim_label, which_snap, rotation,
                 band='sdss_r', derotate=True, width=50, resolution=400):
        self.sim_label = sim_label
        self.which_snap = which_snap
        sim_name = SIM_DICT[sim_label]
        self.sim_name = sim_name
        self.prefix = f'{sim_label}_{which_snap:04}'
        self.band = band

        self.snap, quat_rot = get_snap(which_snap, sim_name, derotate)

        self.width = width #st.sidebar.slider('Width (kpc)', min_value=1, max_value=80, value=50, step=5)
        self.resolution = resolution
        # st.sidebar.slider('Resolution', min_value=100, max_value=500, value=200, step=100)
        tbl_traj = get_traj(sim_name)
        self.tbl_traj = tbl_traj

        self.x, self.y, self.z, self.cur_pos, self.cur_vel, self.r, self.v = get_trajectory(tbl_traj, which_snap)
        self.peri_idx = get_peri_idx(tbl_traj)
        orbital_position = np.array(self.cur_pos)
        orbital_velocity = np.array(self.cur_vel)

        self.rm = rotation.as_matrix()

        pynbody.analysis.halo.center(self.snap.s, vel=False)

        tr = pynbody.transformation.GenericRotation(self.snap, self.rm)

        self.orbital_position_rotated = rotation.apply(orbital_position)
        self.orbital_velocity_rotated = rotation.apply(orbital_velocity)
        self.r_xy = np.linalg.norm(self.orbital_position_rotated[0:2])
        self.los_velocity = -self.orbital_velocity_rotated[2]  # minus for astronomical convention

        self.time = tbl_traj["t"][which_snap]
        self.time_since_peri = self.time - tbl_traj["t"][self.peri_idx[0]]
        self.time_since_2nd_peri = self.time - tbl_traj["t"][self.peri_idx[1]]
        self.tau = self.tbl_traj['t_period'][which_snap]



class FigMaps(PreparedSnap):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fig_sb, self.ax_sb = plt.subplots(dpi=DPI)
        self.fig_hi, self.ax_hi = plt.subplots(dpi=DPI)
        self.fig_hi_kin, self.ax_hi_kin = plt.subplots(dpi=DPI)
        self.fig_star_vel, self.ax_star_vel = plt.subplots(dpi=DPI)

    @lru_cache(1)
    def get_hi_img(self, width=None):
        width = width if width is not None else self.width
        return create_hi_image(self.snap, self.ax_hi, width=width, resolution=self.resolution, ret_im=True)

    @lru_cache(1)
    def get_sb_img(self, width=None):
        width = width if width is not None else self.width
        sb_range = st.sidebar.slider('SB range', min_value=19.0, max_value=30.0, value=(21.0, 27.5), step=0.5)
        return create_fig_sb(self.snap, self.ax_sb, sb_range, self.band, width, self.resolution)

    def _get_kinematics(self, width=None, minimum_sb=None):
        width = width if width is not None else self.width
        star_v_range = st.sidebar.slider('Star v range', min_value=-70, max_value=70, value=(-50, 50), step=5)
        return create_stellar_velocity_map(self.snap,
            self.get_sb_img(width),
            self.ax_star_vel,
            star_v_range,
            width, self.resolution,
            minimum_sb=minimum_sb,
            astronomical_convention=True)

    @lru_cache(1)
    def _get_hi_kinematics(self, width=None):
        width = width if width is not None else self.width
        hi_v_range = (None, None)
        #st.sidebar.slider('Star v range', min_value=-70, max_value=70, value=(-50, 50), step=5)
        return create_hi_vel_image(self.snap,
            self.ax_hi_kin,
            self.get_hi_img(width),
            hi_v_range,
            5e+18, self.rm,
            width, self.resolution,
            astronomical_convention=True)

    def sb_map(self, pdfcrop=True):
        self.get_sb_img()
        self.ax_sb.set_title(rf'$\mu_{{{self.band}}}$ - ({self.which_snap}) - {self.time:.2f} Gyr')

        add_cluster_center_velocity_directions(self.ax_sb,
            self.orbital_position_rotated,
            self.orbital_velocity_rotated)

        do_hi_contours(self.get_hi_img(), levels_hi, self.ax_sb, self.width)
        file_name = f'{self.prefix}_sb.pdf'
        self.fig_sb.savefig(file_name)
        if pdfcrop:
            os.system(f'pdfcrop {file_name}')
        return self.ax_sb

    def kinematics_map(self, width=None, minimum_sb=None, pdfcrop=True):
        self._get_kinematics(width=width, minimum_sb=minimum_sb)

        # do_hi_contours(self.get_hi_img(), levels_hi, self.ax_star_vel, self.width)

        file_name = f'{self.prefix}_kin.pdf'
        self.fig_star_vel.savefig(file_name)
        if pdfcrop:
            os.system(f'pdfcrop {file_name}')
        return self.ax_star_vel

    def hi_kinematics_map(self, width=None, pdfcrop=True):
        self._get_hi_kinematics(width=width)

        # do_hi_contours(self.get_hi_img(), levels_hi, self.ax_star_vel, self.width)

        file_name = f'{self.prefix}_hi_kin.pdf'
        self.fig_hi_kin.savefig(file_name)
        if pdfcrop:
            os.system(f'pdfcrop {file_name}')
        return self.ax_hi_kin


if __name__ == '__main__':
    maps = FigMaps('69p100', which_snap=95, rot=(135, 110, -115))
    maps.ax_sb.grid(True, ls=':')
    maps.sb_map()
    maps.ax_star_vel.grid(True, ls=':')
    maps.kinematics_map(width=6, minimum_sb=24)
    maps.ax_hi_kin.grid(True, ls=':')
    maps.hi_kinematics_map(width=60)
