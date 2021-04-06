from collections import defaultdict

import numpy as np
import pandas as pd
from angles_generator import AnglesGenerator
import tqdm


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
            ag = AnglesGenerator(sim_label, s+1, _rp, _vp, sign, sol)
            d[f'phi'].append(ag.phi)
            d[f'theta'].append(ag.theta)

    return d

def compute_quest(sim_label):
    print('Computing solutions for', sim_label)
    length = 445 if sim_label.startswith('41') else 563
    d = generate_table(sim_label, length)
    df = pd.DataFrame(d)
    df.to_pickle(f'../quest/sol/{sim_label}.sol')
    from astropy.table import Table
    tbl = Table.from_pandas(df)
    tbl.write(f'../quest/sol/{sim_label}.sol.fits', overwrite=True)


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
