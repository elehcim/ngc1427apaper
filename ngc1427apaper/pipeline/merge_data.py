import os
import glob
import pandas as pd
from simulation.simdata import SIM_NAME_DICT, SPECIAL_DICT
import numpy as np
SIM_DICT = {**SIM_NAME_DICT, **SPECIAL_DICT}

files_folder='quest/iso_hi'

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
dff = pd.read_pickle('cache_with_iso.pkl').reset_index(drop=True)
# dff_reindex = dff.reset_index(drop=True)

# Get only oriented snapshots
with_sol = dff.index[dff.solution_found]

# Add to them only the hi_max column
# IMPORTANT: Use array otherwise it assigns looking for the index
assert len(dff.loc[with_sol]) == len(dfh)
dff.loc[with_sol, 'hi_max'] = dfh.hi_max.array

isophote_target = 21.5, 22.5, 23.5, 24.5, 25.5

# From data cache1:
# -44 < theta_sb < 224
# -90 < theta_hi < 90

# theta_hi_sanitized is in [-180, 0]
dff.loc[with_sol, 'theta_hi_sanitized'] = np.where(dfh['theta_hi']>0, dfh['theta_hi']-180, dfh['theta_hi'])


for i in isophote_target:
    dfh[f'a{i}_sanitized'] = np.where(dfh[f'a{i}'] > dfh[f'b{i}'], dfh[f'a{i}'], dfh[f'b{i}'])
    dfh[f'b{i}_sanitized'] = np.where(dfh[f'a{i}'] < dfh[f'b{i}'], dfh[f'a{i}'], dfh[f'b{i}'])
    dff.loc[with_sol, f'xc{i}'] = dfh[f'xc{i}'].array
    dff.loc[with_sol, f'yc{i}'] = dfh[f'yc{i}'].array
    dff.loc[with_sol, f'a{i}'] = dfh[f'a{i}_sanitized'].array
    dff.loc[with_sol, f'b{i}'] = dfh[f'b{i}_sanitized'].array
    dff.loc[with_sol, f'theta_sb{i}'] = dfh[f'theta{i}'].array
    dff[f'e{i}'] = np.sqrt(1-dff[f'b{i}']**2/dff[f'a{i}']**2)

# In this way theta_sb_sanitized is in [-90, 90]
for i in isophote_target:
    dff[f'theta_sb{i}_sanitized'] = np.where(dff[f'theta_sb{i}']>90, dff[f'theta_sb{i}']-180, dff[f'theta_sb{i}'])
    dff[f'alpha{i}'] = 30 - dff[f'theta_sb{i}_sanitized']
    dff[f'beta{i}'] = dff[f'theta_sb{i}_sanitized'] - dff['theta_hi_sanitized']


# to_rename_columns = 'theta_sb', 'alpha', 'beta', 'xc', 'yc', 'a', 'b', 'e'

# new_col_names = dict()
# for col in to_rename_columns:
#     for i, iso in enumerate((26.0, 26.5, 27.0)):
#         new_col_names[f'{col}{i}'] = f'{col}{iso}'

# sanitized = {'theta_sb0_sanitized':'theta_sb26.0_sanitized',
#              'theta_sb1_sanitized':'theta_sb26.5_sanitized',
#              'theta_sb2_sanitized':'theta_sb27.0_sanitized'
#             }

# new_col_names.update(sanitized)

# print(new_col_names)
# dff.rename(columns=new_col_names, inplace=True)


# TODO drop some columns

outname = 'cache_with_multi_iso_and_hi_valid.pkl'
print(f'Writing: {outname}')
dff.to_pickle(outname)

from astropy.table import Table
tbl = Table.from_pandas(dff).filled(np.nan)
tbl.write(f'{outname}.fits', overwrite=True)
