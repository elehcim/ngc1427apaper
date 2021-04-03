from collections import defaultdict
from collections.abc import Iterable

import numpy as np
import pandas as pd
import tqdm
from simulation.ellipse_fit import fit_contour, NoIsophoteFitError

from ngc1427apaper.helper import get_hi, get_sb
from mappers import MapperFromTable
from morphological_quest_script_valid_hi import AnglesGeneratorFromTable, chunks

def compute_multi_iso(dff, idxs):

    isophote_target = 21.5, 22.5, 23.5, 24.5, 25.5
    d = defaultdict(list)
    for i in tqdm.tqdm(idxs):
        row = dff.iloc[i]
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
    return d


def compute_quest(dff, chunk, size):
    length = len(dff)
    lst = tuple(chunks(range(length), size))[chunk]
    # limits = 684*4, 689*4
    # lst = range(*limits)
    print(f'Computing chunk {chunk} from {lst[0]} to {lst[-1]} of {length}')
    d = compute_multi_iso(dff, lst)
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
    stem = f'../quest/bright_iso/iso.{chunk:02}'
    df.to_pickle(f'{stem}.pkl')
    from astropy.table import Table
    tbl = Table.from_pandas(df)
    tbl.write(f'{stem}.fits', overwrite=True)


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
    compute_quest(dff, args.chunk, args.size)

if __name__ == '__main__':
    main()

