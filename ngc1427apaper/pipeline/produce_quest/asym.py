import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
from ngc1427apaper.helper import get_hi, get_sb
from mappers import MapperFromTable
import statmorph
from simulation import derived
from simulation.luminosity import SUN_ABS_MAGNITUDES

def morph_to_df(morph, index=[0]):
    items = morph.__dict__.copy()
    for k in morph.__dict__.keys():
        if k.startswith('_'):
            items.pop(k)
    items.pop('label')
    df = pd.DataFrame(items, index=index)
    return df

_EMPTY_MORPH = morph_to_df(statmorph.SourceMorphology(np.ones((2,2)),np.ones((2,2)),1, gain=1))
_NAN_MORPH = pd.DataFrame(np.zeros((1,len(_EMPTY_MORPH.columns)))*np.nan, columns=_EMPTY_MORPH.columns)


class NonParametricMeasuresFromTable(MapperFromTable):
    def __init__(self, row, width=35, resolution=200):
        self.width = width
        self.resolution = resolution
        super().__init__(row)

    def get_hi(self, ellipse_smooth=(10, 10)):
        hi_img = get_hi(self.snap, self.width, self.resolution, ellipse_smooth=ellipse_smooth)
        return np.min(hi_img), np.max(hi_img)

    def get_lum(self, band='sdss_r', lum_range=(None, None)):
        lum = get_sb(self.snap, lum_range, band, self.width, self.resolution, lum_pc2=True)
        return lum


def compute_morph(dff, idxs, resolution, I_limit, **kwargs):
    l = list()
    for i in tqdm.tqdm(idxs):
        row = dff.iloc[i]
        # print(row.sim, row.snap)
        # print(row.sim, row.snap)
        ag = NonParametricMeasuresFromTable(row, resolution=resolution)
        ag.process()
        image = ag.get_lum()
        segmap = np.ones((resolution, resolution), dtype=np.int32)
        segmap[image<I_limit] = 0

        if segmap.max() == 0:
            l.append(_NAN_MORPH)
            continue
        gain = 1000
        params = statmorph.source_morphology(image, segmap=segmap, gain=gain, **kwargs)
        morph = params[0]
        l.append(morph_to_df(morph, index=[i]))
    d = pd.concat(l)
    return d

# from https://stackoverflow.com/a/312464
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def compute_quest(dff, chunk, size, resolution=200, mu_limit=27):
    length = len(dff)
    lst = tuple(chunks(range(length), size))[chunk]
    # limits = 684*4, 689*4
    # lst = range(*limits)
    # lst = tuple(range(2000+1533, 2000+1540))
    print(f'Computing chunk {chunk} from {lst[0]} to {lst[-1]} of {length}')
    M_sun_sdss_r = SUN_ABS_MAGNITUDES['sdss_r']
    # mu_limit = 27
    I_limit = 10**(0.4*(M_sun_sdss_r + 21.572 - mu_limit))

    df = compute_morph(dff, lst, resolution, I_limit, skybox_size=8)


    # print(dff['sim'].iloc[slice(*limits)])
    # print(len(dff['sim'].iloc[slice(*limits)]))
    # print(df)
    # IMPORTANT: Use array otherwise it assigns looking for the index
    # df['sim'] = dff['sim'].iloc[slice(*limits)].array
    # df['snap'] = dff['snap'].iloc[slice(*limits)].array
    # breakpoint()
    # df['t_period'] = dff['t_period'].iloc[slice(*limits)].array
    # print(dff[['sim', 'snap']].iloc[slice(*limits)])
    stem = f'../quest/morph/morph.{chunk:02}'
    df.to_pickle(stem + '.pkl')
    from astropy.table import Table
    tbl = Table.from_pandas(df)
    tbl.write(stem + '.fits', overwrite=True)


def parse_args(cli=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='chunk', help="Chunk idx", type=int)
    parser.add_argument('--cache-file', help="Cache file", default='cache_with_multi_iso_and_hi_valid.pkl')
    # parser.add_argument('--cache-file', help="Cache file", default='cache_with_iso.pkl')
    parser.add_argument('-s', dest='size', help="Chunk size", default=2000, type=int)
    parser.add_argument('-r', dest='resolution', help="Image resolution", default=200, type=int)
    parser.add_argument('-m', '--mu', dest='mu_limit', help="SB limit for background", default=27, type=float)
    args = parser.parse_args(cli)
    return args

def main(cli=None):
    args = parse_args(cli)
    rx, ry = np.cos(np.pi/3), np.sin(np.pi/3)
    dff = pd.read_pickle(args.cache_file).query(f'not phi.isnull() & hi_max > 1e15 & (vx_rot * {rx} + vy_rot * {ry})>0 & not sim.str.endswith("r")')
    compute_quest(dff, chunk=args.chunk, size=args.size, resolution=args.resolution, mu_limit=args.mu_limit)


def print_morph(morph):
    print('xc_centroid =', morph.xc_centroid)
    print('yc_centroid =', morph.yc_centroid)
    print('ellipticity_centroid =', morph.ellipticity_centroid)
    print('elongation_centroid =', morph.elongation_centroid)
    print('orientation_centroid =', morph.orientation_centroid)
    print('xc_asymmetry =', morph.xc_asymmetry)
    print('yc_asymmetry =', morph.yc_asymmetry)
    print('ellipticity_asymmetry =', morph.ellipticity_asymmetry)
    print('elongation_asymmetry =', morph.elongation_asymmetry)
    print('orientation_asymmetry =', morph.orientation_asymmetry)
    print('rpetro_circ =', morph.rpetro_circ)
    print('rpetro_ellip =', morph.rpetro_ellip)
    print('rhalf_circ =', morph.rhalf_circ)
    print('rhalf_ellip =', morph.rhalf_ellip)
    print('r20 =', morph.r20)
    print('r80 =', morph.r80)
    print('Gini =', morph.gini)
    print('M20 =', morph.m20)
    print('F(G, M20) =', morph.gini_m20_bulge)
    print('S(G, M20) =', morph.gini_m20_merger)
    print('sn_per_pixel =', morph.sn_per_pixel)
    print('C =', morph.concentration)
    print('A =', morph.asymmetry)
    print('S =', morph.smoothness)
    print('sersic_amplitude =', morph.sersic_amplitude)
    print('sersic_rhalf =', morph.sersic_rhalf)
    print('sersic_n =', morph.sersic_n)
    print('sersic_xc =', morph.sersic_xc)
    print('sersic_yc =', morph.sersic_yc)
    print('sersic_ellip =', morph.sersic_ellip)
    print('sersic_theta =', morph.sersic_theta)
    print('sky_mean =', morph.sky_mean)
    print('sky_median =', morph.sky_median)
    print('sky_sigma =', morph.sky_sigma)
    print('flag =', morph.flag)
    print('flag_sersic =', morph.flag_sersic)

if __name__ == '__main__':
    main()




    # args = parse_args()

    # from simulation.luminosity import SUN_ABS_MAGNITUDES
    # M_sun_sdss_r = SUN_ABS_MAGNITUDES['sdss_r']
    # mu_limit = 27
    # I_limit = 10**(0.4*(M_sun_sdss_r + 21.572 - mu_limit))

    # resolution = 500

    # print(f"{I_limit} L_sun/pc^2")




    # dff = pd.read_pickle(args.cache_file)
    # row = dff.query('sim == "68p100" & snap == 49 & rp == 77 & vp == -793 & sol == 2 & sign == 1').iloc[0]
    # ag = NonParametricMeasuresFromTable(row, resolution=resolution)
    # ag.process()
    # image = ag.get_lum()
    # segmap = np.ones((resolution,resolution), dtype=np.int32)
    # segmap[image<I_limit] = 0
    # # plt.imshow(np.log10(image), origin='lower')
    # # plt.colorbar()
    # gain = 1
    # params = statmorph.source_morphology(image, segmap=segmap, gain=gain)
    # morph = params[0]
    # from statmorph.utils.image_diagnostics import make_figure
    # fig = make_figure(morph)
    # print_morph(morph)
    # df = morph_to_df(morph)
    # plt.show()


    # my_asym =
    # b = image[~segmap]
    #