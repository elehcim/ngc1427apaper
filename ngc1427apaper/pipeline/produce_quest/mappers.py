import numpy as np
from simulation.simdata import SIM_NAME_DICT, SPECIAL_DICT
import pynbody
from ngc1427apaper.helper import phitheta2rotation, get_snap

SIM_DICT = {**SIM_NAME_DICT, **SPECIAL_DICT}


class MapperFromTable:
    def __init__(self, row, width=35, resolution=200):
        self.snap = None
        self._rotation = None
        if np.isnan(row.phi):
            raise ValueError(f'No solution for this row')
        self.sim_name = SIM_DICT[row.sim]
        self.which_snap = row.snap
        self.width = width
        self.resolution = resolution
        self._orbital_position = np.array([row.x, row.y, row.z])
        # self._orbital_velocity = np.array([row.vx, row.vy, row.vz])

        self.phi, self.theta = row.phi, row.theta

    def process(self):
        self._rotation = phitheta2rotation(self.phi, self.theta, self._orbital_position)
        rotation_matrix = self._rotation.as_matrix()
        derotate = True
        snap, quat_rot = get_snap(self.which_snap, self.sim_name, derotate)
        pynbody.analysis.halo.center(snap.s, vel=False)
        tr = pynbody.transformation.GenericRotation(snap, rotation_matrix)
        self.snap = snap
