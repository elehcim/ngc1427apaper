import os

import numpy as np
import pandas as pd
import glob
from ngc1427apaper.helper import get_peri_idx
from simulation.simdata import SIM_NAME_DICT, SPECIAL_DICT, get_traj
from scipy.spatial.transform import Rotation as R

_SIM_DICT = {**SIM_NAME_DICT, **SPECIAL_DICT}
def peri(x):
    name = x[3:]
    if name.endswith('r'):
        name = name[0:-1]
    return int(name)

def _compose_quat(p, q):
    product = np.empty((max(p.shape[0], q.shape[0]), 4))
    product[:, 3] = p[:, 3] * q[:, 3] - np.sum(p[:, :3] * q[:, :3], axis=1)
    product[:, :3] = (p[:, None, 3] * q[:, :3] + q[:, None, 3] * p[:, :3] +
                      np.cross(p[:, :3], q[:, :3]))
    return product


def phitheta2rotation_array(phi, theta, orbital_position=None, offset_deg=210.0):
    """The input phi and theta should be in deg"""
    r1 = R.from_rotvec((phi[np.newaxis]*np.pi/180 * np.array([0, 0, 1])[:,np.newaxis]).T)
    y_prime = r1.apply(np.array([0, 1, 0]))
    r2 = R.from_rotvec((theta[np.newaxis]*np.pi/180 * y_prime.T).T)

    axis_rotation = (r2 * r1).inv()
    if orbital_position is not None:
        orbital_position_rotated_plane = axis_rotation.apply(orbital_position)
        z_rot = np.arctan2(orbital_position_rotated_plane[:,1],
                           orbital_position_rotated_plane[:,0])
        # Add an offset of 210 degree to fix the cluster center direction to north-east
        _value_z_rot = (-z_rot*180.0/np.pi+offset_deg) % 360.0
        r3 = R.from_euler('z', _value_z_rot, degrees=True)
        rotation = r3 * axis_rotation
    else:
        rotation = axis_rotation
    return rotation


def get_dff(files_folder):
    sol2label = lambda f: os.path.splitext(os.path.basename(f))[0]

    files = glob.glob(os.path.join(files_folder,'*.sol'))
    labels = [sol2label(f) for f in files]
    df_list = list()

    for s in _SIM_DICT.keys():
        if s in labels:
            dfl = pd.read_pickle(f'{files_folder}/{s}.sol')
            sim_name = _SIM_DICT[s]
            dfl['sim'] = s
            dfl['peri'] = peri(s)
            traj = get_traj(sim_name).to_pandas()
            peri_idx1, peri_idx2 = get_peri_idx(traj)
            print(s, peri_idx1, peri_idx2)
            rep = len(dfl)/len(traj)
            dfl['t_period'] = traj['t_period'].repeat(rep).reset_index(drop=True)
            dfl['t'] = traj['t'].repeat(rep).reset_index(drop=True)
            dfl['tp1'] = (traj['t']-traj['t'][peri_idx1]).repeat(rep).reset_index(drop=True)
            dfl['tp2'] = (traj['t']-traj['t'][peri_idx2]).repeat(rep).reset_index(drop=True)

            dfl['orbital_phase'] = traj['orbital_phase'].repeat(rep).reset_index(drop=True)
            dfl['offset_orbital_phase'] = traj['offset_orbital_phase'].repeat(rep).reset_index(drop=True)
            # Because some trajectories contain nans especially at the end (e.g. 71p100)
            traj_clean = traj.fillna(method='ffill')
            # print(traj_clean.info())
            orbital_position = np.array([traj_clean['x'], traj_clean['y'], traj_clean['z']]).T.repeat(rep, axis=0)
            orbital_velocity = np.array([traj_clean['vx'], traj_clean['vy'], traj_clean['vz']]).T.repeat(rep, axis=0)
            dfl['phi_clean'] = dfl['phi'].fillna(0)
            dfl['theta_clean'] = dfl['theta'].fillna(0)
            dfl[['x', 'y', 'z']] = orbital_position
            dfl[['vx', 'vy', 'vz']] = orbital_velocity
            solution_found = ~np.isnan(dfl['phi'].to_numpy())
            rotation = phitheta2rotation_array(dfl['phi_clean'].to_numpy(),
                                               dfl['theta_clean'].to_numpy(),
                                               orbital_position)

            # orbital_position_rotated = rotation.apply(orbital_position)
            # orbital_velocity_rotated = rotation.apply(orbital_velocity)
            dfl['solution_found'] = solution_found
            orbital_position_rotated = np.where(solution_found[:, np.newaxis],
                                                rotation.apply(orbital_position),
                                                np.ones((1,3))*np.nan)
            orbital_velocity_rotated = np.where(solution_found[:, np.newaxis],
                                                rotation.apply(orbital_velocity),
                                                np.ones((1,3))*np.nan)
            dfl[['x_rot', 'y_rot', 'z_rot']] = orbital_position_rotated
            dfl[['vx_rot', 'vy_rot', 'vz_rot']] = orbital_velocity_rotated

            dfl['theta_pos'] = np.arctan2(orbital_position_rotated[:, 1],
                                          orbital_position_rotated[:, 0]) * 180/np.pi

            df_list.append(dfl)

    dff = pd.concat(df_list)
    dff['rp_found'] = np.linalg.norm(dff[['x_rot', 'y_rot']], axis=1)
    dff['nearest_peri'] = np.where(np.abs(dff['tp1']) < np.abs(dff['tp2']), 1, 2 )
    return dff, orbital_position_rotated

if __name__ == '__main__':
    dff, pos = get_dff(files_folder='quest/sol/')
    # From data cache1:
    # -44 < theta_sb < 224
    # -90 < theta_hi < 90

    # In this way theta_sb_sanitized is in [-90, 90]
    # theta_hi_sanitized is in [-180, 0]
    # dff['theta_hi_sanitized'] = np.where(dff['theta_hi']>0, dff['theta_hi']-180, dff['theta_hi'])
    # for i in range(3):
    #     dff[f'theta_sb{i}_sanitized'] = np.where(dff[f'theta_sb{i}']>90, dff[f'theta_sb{i}']-180, dff[f'theta_sb{i}'])
    #     # print(dff[['alpha', 'beta', 'theta_sb', 'theta_hi', 'theta_sb_sanitized', 'theta_hi_sanitized']].describe())
    #     dff[f'alpha{i}'] = 30 - dff[f'theta_sb{i}_sanitized']
    #     dff[f'beta{i}'] = dff[f'theta_sb{i}_sanitized'] - dff['theta_hi_sanitized']
    #     print(dff[[f'alpha{i}', f'beta{i}', f'theta_sb{i}_sanitized', 'theta_hi_sanitized']].describe())

    dff.to_pickle('cache_with_iso.pkl')
