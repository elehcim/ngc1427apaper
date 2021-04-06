import numpy as np
from simulation.ellipse_fit import fit_contour, NoIsophoteFitError, EllParams
from hi_tail import get_aperture_from_moments

from ngc1427apaper.helper import get_hi, get_sb
from mappers import MapperFromTable
from collections.abc import Iterable
from photutils import EllipticalAperture


def sanitize_ellipseparams(ell):
    xc, yc = ell[0:2]
    a, b = ell[2:4]
    theta = ell[4]
    if a < b:
        a, b = b, a
        theta += np.pi/2
    if theta > 2*np.pi:
        theta = theta % 2*np.pi
    return EllParams(xc, yc, a, b, theta)

def get_iterable(x):
    if isinstance(x, Iterable):
        return x
    else:
        return (x,)


class AnglesGeneratorFromTable(MapperFromTable):
    def __init__(self, row, width=35, resolution=200):
        super().__init__(row)
        self.width = width
        self.resolution = resolution

    def get_hi(self, ellipse_smooth=(10, 10)):
        hi_img = get_hi(self.snap, self.width, self.resolution, ellipse_smooth=ellipse_smooth)
        return np.min(hi_img), np.max(hi_img)

    def get_params_from_sb(self, isophote_sb, band='sdss_r', sb_range=(21.0, 27.5) ):
        sb = get_sb(self.snap, sb_range, band, self.width, self.resolution)
        self._isophote_sb = get_iterable(isophote_sb)
        params = list()
        for iso in self._isophote_sb:
            try:
                # 'xc', 'yc', 'a', 'b', 'theta'
                ell_fitted = fit_contour(sb, iso, self.width, self.resolution)
                ell = sanitize_ellipseparams(ell_fitted)

                # if not np.isnan(print(ell)
                # if ell.a < ell.b:
                #     tmp = ell.a
                #     ell.a = ell.b
                #     ell.b = ell.a
                # if ell.theta > np.pi/2:
                #     ell.theta -= np.pi
                # ell = np.array([xc, yc, a, b, theta])
            except (ValueError, NoIsophoteFitError):
                ell = np.array([np.nan]*5)
            params.append(ell)

        return np.array(params)

    def get_angle_from_hi(self, ellipse_smooth=(10, 10)):
        hi_img = get_hi(self.snap, self.width, self.resolution, ellipse_smooth=ellipse_smooth)
        aperture_hi = get_aperture_from_moments(hi_img, self.width, self.resolution)
        self.theta_hi = aperture_hi.theta
        return aperture_hi.theta
