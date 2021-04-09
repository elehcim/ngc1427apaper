import os
import astropy
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import numpy as np
import matplotlib.pyplot as plt
import aplpy
from ngc1427apaper.helper import savefig

# # f = fits.open('ADP.2020-08-26T11:45:32.273_crop.fits')
# # hdu = f[0]
# # header = fits.getheader('ADP.2020-08-26T11:45:32.273.fits')
# # fig, ax = plt.subplots()


ra, dec = 55.04, -35.62
def cutout(image):

    # ra, dec = 55.04, -35.62
    hdu, hdr = fits.getdata(image), fits.getheader(image)
    w = WCS(hdr)
    print(w)
    # find pixelscale
    # pixel_scale = 0.2 / 3600 # deg #3600#find_pixel_scale(hdr)

    center = SkyCoord(ra, dec, unit='deg', frame='icrs')
    ndim = 1800# pixel_scale * 2/60 # round(np.sqrt(min_shape**2 / pixel_scale))
    shape = (ndim, ndim)

    co1 = Cutout2D(hdu, center, shape, wcs=w)
    # create a new FITS HDU
    nhdu = fits.PrimaryHDU(data=co1.data, header=co1.wcs.to_header())
    nhdr = nhdu.header

    pixel_scale = 0.2
    print(- 2.5*np.log10(1/pixel_scale**2))
    nhdu.data = -2.5 *np.log10(nhdu.data) - 2.5*np.log10(1/pixel_scale**2)
    img_name = os.path.splitext(image)[0] + '_mag_cutout.fits' # FIXME
    # write to disk
    nhdu.writeto(img_name, overwrite=True)
    return img_name

img_orig = "ADP.2020-08-26T11:45:32.273.fits"
img_name = cutout(img_orig)

# img_name = 'ADP.2020-08-26T11:45:32.273_mag_cutout.fits'

def get_percentile_values(data, p1, p2):
    return np.percentile(np.unique(data), p1), np.percentile(np.unique(data), p2)

MAG_LIMIT = 27.5
SMOOTH = None
fig = plt.figure()
f = aplpy.FITSFigure(img_name, figure=fig)
print(np.nanmax(f._data))
# print(np.nanmax(f._data))
# f._data[f._data>26] = np.nan
# vmin, vmax = get_percentile_values(f._data, 5, 95)
f.show_colorscale(cmap='binary', vmax=MAG_LIMIT, smooth=SMOOTH)
major_axis = 0.04
fac = 4200/7420  # from Lee Waddell email)
f.show_ellipses(ra, dec-0.002, major_axis, major_axis*fac, angle=-15)

# f._data[f._data>27.5] = np.nan
# f.show_contour()
f.set_nan_color('black')
# f.recenter(55.04,-35.62, radius=2/60)
f.add_colorbar()
f.colorbar.set_location('top')
f.ax.tick_params(direction='in')
f.colorbar._colorbar.ax.tick_params(direction='in')
savefig(fig, f'r_band_{MAG_LIMIT}', ext=".png", dpi=200)

# plt.show()
