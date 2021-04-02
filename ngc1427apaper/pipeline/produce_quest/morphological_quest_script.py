from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynbody
from ngc1427apaper.helper import normify, pos_vel, get_solved_rotation, get_hi, get_snap, get_sb
from functools import lru_cache
from simulation import derived
from simulation.ellipse_fit import fit_contour, ellipseparams2patch, NoIsophoteFitError
from simulation.simdata import SIM_NAME_DICT, SPECIAL_DICT, get_tables, get_magnitudes, get_traj
from hi_tail import get_aperture_from_moments
import tqdm

SIM_DICT = {**SIM_NAME_DICT, **SPECIAL_DICT}
def ellipseparams2aperture(params, **kwargs):
    from photutils import EllipticalAperture
    center = params[0:2]
    a, b = params[2:4]
    # a, b = np.max(ab), np.min(ab)
    # a, b = ab
    theta = params[4]
    if a < b:
        a, b = b, a
        theta += np.pi/2
    if theta > 2*np.pi:
        theta = theta % 2*np.pi
    apertures = EllipticalAperture(center, a, b, theta=theta)
    return apertures

from collections.abc import Iterable
def get_iterable(x):
    if isinstance(x, Iterable):
        return x
    else:
        return (x,)

def get_angle(p0, p1):
    # np.arccos result is in [0, pi]
    angle = np.arccos(np.dot(p0, p1)/(np.linalg.norm(p0) * np.linalg.norm(p1)))
    return np.degrees(angle)


def get_alpha_beta(orbital_position_rotated, theta_hi, theta_sb):
    fac = 1
    p = -normify(orbital_position_rotated, fac)[0:2]
    # v = normify(orbital_velocity_rotated, fac)[0:2]
    h = np.cos(theta_hi), np.sin(theta_hi)
    s = np.cos(theta_sb), np.sin(theta_sb)
    alpha = get_angle(p, s)
    beta = get_angle(h, s)

    return alpha, beta


class NoSolutionError(Exception):
    pass


@lru_cache(32)
def cached_get_traj(sim_name):
    return get_traj(sim_name)


class Mapper:
    """docstring for Mapper"""
    def __init__(self, sim_label, which_snap, rp, vp, sign_of_r, which_solution, width=35, resolution=200):
        self.sim_name = SIM_DICT[sim_label]
        self.which_snap = which_snap
        self.width = width
        self.resolution = resolution
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
        orbital_velocity_rotated = self._rotation.apply(self._orbital_velocity)
        derotate = True
        snap, quat_rot = get_snap(self.which_snap, self.sim_name, derotate)
        pynbody.analysis.halo.center(snap.s, vel=False)
        tr = pynbody.transformation.GenericRotation(snap, rotation_matrix)
        self.snap = snap
        self.orbital_position_rotated = orbital_position_rotated

class AnglesGenerator(Mapper):
    def __init__(self, sim_label, which_snap, rp, vp, sign_of_r, which_solution, width=35, resolution=200):
        super().__init__(sim_label, which_snap, rp, vp, sign_of_r, which_solution, width, resolution)
        # self.isophote_sb = get_iterable(isophote_sb)

    def get_angle_from_sb(self, band='sdss_r', sb_range=(21.0, 27.5), isophote_sb=26.5):
        sb = get_sb(self.snap, sb_range, band, self.width, self.resolution)
        self._isophote_sb = get_iterable(isophote_sb)
        theta_sb = list()
        for iso in self._isophote_sb:
            try:
                ell = fit_contour(sb, iso, self.width, self.resolution)
                t = ellipseparams2aperture(ell).theta
            except (ValueError, NoIsophoteFitError):
                t = np.nan
            theta_sb.append(t)
            # return list(np.full_like(self._isophote_sb, np.nan))

        # self.theta_sb = aperture_sb.theta
        return np.array(theta_sb)

    def get_angle_from_hi(self, ellipse_smooth=(10, 10)):
        hi_img = get_hi(self.snap, self.width, self.resolution, ellipse_smooth=ellipse_smooth)
        aperture_hi = get_aperture_from_moments(hi_img, self.width, self.resolution)
        self.theta_hi = aperture_hi.theta
        return aperture_hi.theta

    # def get_alpha_beta(self, band='sdss_r', sb_range=(21.0, 27.5), ellipse_smooth=(10, 10)):
    #     theta_sb = self.get_angle_from_sb(band, sb_range)
    #     theta_hi = self.get_angle_from_hi(ellipse_smooth)
    #     return get_alpha_beta(self.orbital_position_rotated, theta_hi, theta_sb)

# def compute_angles(sim_label, which_snap, rp, vp, sign_of_r, which_solution):
#     ag = AnglesGenerator(sim_label, which_snap, rp, vp, sign_of_r, which_solution)
#     alpha, beta = ag.get_alpha_beta()
#     return alpha, beta

def generate_table(sim_label, length):
    from itertools import product, chain
    rp, vp = 137, -693
    dr, dv = 60, 100
    isophote_sb = 26.5
    delta_iso = 0.5

    isophote_target = (isophote_sb-delta_iso, isophote_sb, isophote_sb+delta_iso)
    nans = np.full_like(isophote_target, np.nan)

    d = defaultdict(list)
    for s in tqdm.tqdm(range(length)):
        for _rp, _vp, sign, sol in chain(product((rp-dr, rp+dr), (vp-dv, vp+dv), (-1,1), (1,2)),
    # product((rp-dr/2, rp+dr/2), (vp-dv/2, vp+dv/2), (-1,1), (1,2), (isophote_sb-delta_iso, isophote_sb+delta_iso)),
                                        product( (rp,), (vp+dv,), (-1,1), (1,2) ),
                                        product( (rp,), (vp-dv,), (-1,1), (1,2) ),
                                        product( (rp+dr,), (vp,), (-1,1), (1,2) ),
                                        product( (rp-dr,), (vp,), (-1,1), (1,2) ),
                                        product( (rp,), (vp,), (-1,1), (1,2))):
            # breakpoint()
            # print(_rp, _vp, sign, sol, iso)
            d['snap'].append(s+1)
            d['sign'].append(sign)
            d['rp'].append(_rp)
            d['vp'].append(_vp)
            d['sol'].append(sol)
            # d['iso'].append(iso)
            ag = AnglesGenerator(sim_label, s+1, _rp, _vp, sign, sol)
            d[f'phi'].append(ag.phi)
            d[f'theta'].append(ag.theta)
            if np.isnan(ag.phi):
                d[f'theta_sb'].append(nans)
                d[f'theta_hi'].append(np.nan)
            else:
                ag.process()

                theta_sb = ag.get_angle_from_sb(isophote_sb=isophote_target)*180/np.pi

                # breakpoint()
                # for i, t in enumerate(theta_sb):
                    # d[f'theta_sb']
                d[f'theta_sb'].append(theta_sb)
                d[f'theta_hi'].append(ag.get_angle_from_hi()*180/np.pi)
        # except (NoSolutionError, ValueError, TypeError):
        # FIXME Investigate TypeError
                # except (NoIsophoteFitError, TypeError):
                # d[f'alpha'].append(np.nan)
                # d[f'beta'].append(np.nan)
                # d[f'theta_sb'].append(nans)
    #             d[f'theta_hi'].append(np.nan)
    return d

# def tidy_up(d, rep):
    # d1 = dict()
    # for k, v in d.items():
    #     if k == 'theta_sb':
    #         d1[k] = np.hstack(v)
    #     arr = np.array(v).repeat(rep)
    #     d1[k] = arr
    # return pd.DataFrame(d1)

def fix_theta(d):
    arr = np.vstack(np.array(d['theta_sb']))
    # arr =
    return arr[:,0], arr[:,1], arr[:,2]


def compute_quest(sim_label):
    print('Computing quest for', sim_label)
    length = 445 if sim_label.startswith('41') else 563
    d = generate_table(sim_label, length)
    d['theta_sb0'], d['theta_sb1'], d['theta_sb2'] = fix_theta(d)
    del d['theta_sb']
    df = pd.DataFrame(d)
    df.to_pickle(f'../quest/with_iso/{sim_label}.sol')
    from astropy.table import Table
    tbl = Table.from_pandas(df)
    tbl.write(f'../quest/with_iso/{sim_label}.sol.fits', overwrite=True)
    # df.plot()
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.scatter(df['rp'], df['vp'])
    # plt.show()


def parse_args(cli=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='sim_label', help="Simulation label")
    args = parser.parse_args(cli)
    return args

def main(cli=None):
    args = parse_args(cli)
    compute_quest(args.sim_label)

if __name__ == '__main__':
    main()
    # sim_label = '71p300'
    # # d = generate_table(sim_label, 10)
    # # df = pd.DataFrame(d)

    # rp, vp = 137, -693
    # dr, dv = 60, 100
    # isophote_sb = 26.5

    # ag = AnglesGenerator(sim_label, 56, rp, vp, 1, 2)
    # ag.process()
    # print('Computing quest for', sim_label)
    # d = main(sim_label)
    # df = pd.DataFrame(d)
    # from astropy.table import Table
    # tbl = Table.from_pandas(df)
    # tbl.write(f'{sim_label}.sol.fits', overwrite=True)
    # df.to_pickle(f'{sim_label}.sol')
    # df.plot()
    # plt.show()
