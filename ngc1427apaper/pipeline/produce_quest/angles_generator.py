import numpy as np
from simulation.ellipse_fit import fit_contour, NoIsophoteFitError, EllParams
from ngc1427apaper.helper import get_hi, get_sb
from mappers import MapperFromTable
from collections.abc import Iterable


def get_aperture_from_moments(img, width, resolution, ax=None):
    from photutils import data_properties, EllipticalAperture
    import astropy.units as u
    # https://photutils.readthedocs.io/en/stable/api/photutils.segmentation.SourceProperties.html
    cat = data_properties(img)
    position = (np.array((cat.xcentroid.value, cat.ycentroid.value)) - resolution/2) * width/resolution
    # semiminor axis: The 1-sigma standard deviation along the semimajor axis of the 2D Gaussian function
    # that has the same second-order central moments as the source.

    a = cat.semimajor_axis_sigma.value * width/resolution
    b = cat.semiminor_axis_sigma.value * width/resolution
    theta = cat.orientation.to(u.rad).value
    apertures = EllipticalAperture(position, a, b, theta=theta)
    if ax is not None:
        apertures.plot(color='#d62728', axes=ax)
    return apertures


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

def _get_iterable(x):
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
        self._isophote_sb = _get_iterable(isophote_sb)
        params = list()
        for iso in self._isophote_sb:
            try:
                # 'xc', 'yc', 'a', 'b', 'theta'
                ell_fitted = fit_contour(sb, iso, self.width, self.resolution)
                ell = sanitize_ellipseparams(ell_fitted)
            except (ValueError, NoIsophoteFitError):
                ell = np.array([np.nan]*5)
            params.append(ell)

        return np.array(params)

    def get_angle_from_hi(self, ellipse_smooth=(10, 10)):
        hi_img = get_hi(self.snap, self.width, self.resolution, ellipse_smooth=ellipse_smooth)
        aperture_hi = get_aperture_from_moments(hi_img, self.width, self.resolution)
        self.theta_hi = aperture_hi.theta
        return aperture_hi.theta
