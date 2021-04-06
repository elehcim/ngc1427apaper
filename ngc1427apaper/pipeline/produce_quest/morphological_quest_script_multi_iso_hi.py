from collections import defaultdict

import numpy as np
import pandas as pd
import tqdm
from angles_generator import AnglesGeneratorFromTable

def compute_hi_extrema_and_iso(dff, idxs):
    isophote_target = 21.5, 22.5, 23.5, 24.5, 25.5, 26.0, 26.5, 27.0

    d = defaultdict(list)
    for c in tqdm.tqdm(idxs):
        row = dff.iloc[c]
        # print(row.sim, row.snap)
        # print(row.sim, row.snap)
        ag = AnglesGeneratorFromTable(row)
        ag.process()
        params = ag.get_params_from_sb(isophote_sb=isophote_target)
        for i, p in zip(isophote_target, params):
            d[f'xc{i}'].append(p[0])
            d[f'yc{i}'].append(p[1])
            d[f'a{i}'].append(p[2])
            d[f'b{i}'].append(p[3])
            d[f'theta{i}'].append(p[4]*180.0/np.pi)
        hi_min, hi_max = ag.get_hi()
        d[f'theta_hi'].append(ag.get_angle_from_hi() * 180 / np.pi)
        d[f'hi_min'].append(hi_min)
        d[f'hi_max'].append(hi_max)
    return d

# from https://stackoverflow.com/a/312464
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def compute_quest_chunk(dff, chunk, size):
    length = len(dff)
    lst = tuple(chunks(range(length), size))[chunk]
    # limits = 684*4, 689*4
    # lst = range(*limits)
    print(f'Computing chunk {chunk} from {lst[0]} to {lst[-1]} of {length}')
    d = compute_hi_extrema_and_iso(dff, lst)
    df = pd.DataFrame(d)
    # print(dff['sim'].iloc[slice(*limits)])
    # print(len(dff['sim'].iloc[slice(*limits)]))
    # print(df)
    # IMPORTANT: Use array otherwise it assigns looking for the index
    # df['sim'] = dff['sim'].iloc[slice(*limits)].array
    # df['snap'] = dff['snap'].iloc[slice(*limits)].array
    # breakpoint()
    # df['t_period'] = dff['t_period'].iloc[slice(*limits)].array
    # print(dff[['sim', 'snap']].iloc[slice(*limits)])
    df.to_pickle(f'../quest/iso_hi/iso_hi.{chunk:02}.pkl')
    from astropy.table import Table
    tbl = Table.from_pandas(df)
    tbl.write(f'../quest/iso_hi/iso_hi.{chunk:02}.fits', overwrite=True)


def parse_args(cli=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='chunk', help="Chunk idx", type=int)
    parser.add_argument('--cache-file', help="Cache file", default='cache_with_iso.pkl')
    parser.add_argument('-s', dest='size', help="Chunk size", default=3000, type=int)
    args = parser.parse_args(cli)
    return args

def main(cli=None):
    args = parse_args(cli)
    dff = pd.read_pickle(args.cache_file).query('not phi.isnull()')
    compute_quest_chunk(dff, args.chunk, args.size)

if __name__ == '__main__':
    main()

