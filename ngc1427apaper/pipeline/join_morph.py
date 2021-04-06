import os
import glob
import pandas as pd
import numpy as np

files_folder='quest/morph'

files = sorted(glob.glob(os.path.join(files_folder,'*.??.pkl')))
print(files)


df_list = list()
for f in files:
    print(f)
    dfl = pd.read_pickle(f)
    df_list.append(dfl)
dfh = pd.concat(df_list).reset_index(drop=True)

print(len(dfh))

# reindex because I have some duplicated entries
dff = pd.read_pickle('cache_with_multi_iso_and_hi_valid.pkl').reset_index(drop=True)

rx, ry = np.cos(np.pi/3), np.sin(np.pi/3)
queried = dff.query(f'not phi.isnull() & hi_max > 1e15 & (vx_rot * {rx} + vy_rot * {ry})>0 & not sim.str.endswith("r")')

# Get only oriented and selected snapshots
good = queried.index

assert len(dff.loc[good]) == len(dfh)
# IMPORTANT: Use array otherwise it assigns looking for the index
for col in dfh.columns:
    print(col)
    dff.loc[good, col] = dfh[col].array


df_filtered = dff.query(f'not phi.isnull() & hi_max > 1e15 & (vx_rot * {rx} + vy_rot * {ry})>0 & not sim.str.endswith("r")')
df_filtered.to_pickle('selected_with_multi_iso_and_morph.pkl')
