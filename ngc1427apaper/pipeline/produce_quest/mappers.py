import numpy as np
import pynbody
from ngc1427apaper.helper import pos_vel, get_solved_rotation, get_snap, phitheta2rotation
from functools import lru_cache
from simulation.simdata import SIM_NAME_DICT, SPECIAL_DICT, get_traj


SIM_DICT = {**SIM_NAME_DICT, **SPECIAL_DICT}


@lru_cache(32)
def cached_get_traj(sim_name):
    return get_traj(sim_name)


class Mapper:
    """docstring for Mapper"""
    def __init__(self, sim_label, which_snap, rp, vp, sign_of_r, which_solution):
        self.sim_name = SIM_DICT[sim_label]
        self.which_snap = which_snap
        tbl_traj = cached_get_traj(self.sim_name)
        self._orbital_position, self._orbital_velocity = pos_vel(tbl_traj, which_snap-1)

        self._rotation, phi, theta = get_solved_rotation(self._orbital_position, self._orbital_velocity,
                                                         rp, vp, which_solution, sign_of_r)
        self.phi, self.theta = phi, theta

    def process(self):
        # if np.isnan(phi):
        #     raise NoSolutionError(f'No solution for {sim_label}:{which_snap}')
        rotation_matrix = self._rotation.as_matrix()

        orbital_position_rotated = self._rotation.apply(self._orbital_position)
        # orbital_velocity_rotated = self._rotation.apply(self._orbital_velocity)
        snap, quat_rot = get_snap(self.which_snap, self.sim_name, derotate=True)
        pynbody.analysis.halo.center(snap.s, vel=False)
        tr = pynbody.transformation.GenericRotation(snap, rotation_matrix)
        self.snap = snap
        self.orbital_position_rotated = orbital_position_rotated


class MapperFromTable:
    def __init__(self, row):
        self.snap = None
        self._rotation = None
        if np.isnan(row.phi):
            raise ValueError(f'No solution for this row')
        self.sim_name = SIM_DICT[row.sim]
        self.which_snap = row.snap
        self._orbital_position = np.array([row.x, row.y, row.z])
        # self._orbital_velocity = np.array([row.vx, row.vy, row.vz])

        self.phi, self.theta = row.phi, row.theta

    def process(self):
        self._rotation = phitheta2rotation(self.phi, self.theta, self._orbital_position)
        rotation_matrix = self._rotation.as_matrix()
        snap, quat_rot = get_snap(self.which_snap, self.sim_name, derotate=True)
        pynbody.analysis.halo.center(snap.s, vel=False)
        tr = pynbody.transformation.GenericRotation(snap, rotation_matrix)
        self.snap = snap
