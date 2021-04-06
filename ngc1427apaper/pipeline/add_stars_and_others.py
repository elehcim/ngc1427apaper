import pandas as pd
import numpy as np
from simulation.simdata import get_tables, SIM_NAME_DICT, get_mach, get_cii
from simulation.units import gadget_dens_units
import tqdm
import pynbody
from collections import defaultdict
from ngc1427apaper.globals import isophote_target, isophote_target_2

table_columns = ['mass_star', 'r_eff3d', 'sfr', 'rho_host']
mach_columns = ['mach', 'temp_host']
cii_columns = ['cii'] # erg s**-1
columns = table_columns + mach_columns + cii_columns

tbl_dict = dict()
for k,v in SIM_NAME_DICT.items():
    _t = get_tables(v, True)[table_columns].to_pandas()
    _m = get_mach(v)[mach_columns].to_pandas()
    _cii = get_cii(v)[cii_columns].to_pandas()
    tbl_dict[k] = pd.concat([_t, _m, _cii], axis=1, sort=False)


# reindex because indexes are relative to the unfiltered df
dff = pd.read_pickle('selected_with_multi_iso_and_morph.pkl').reset_index(drop=True)
# query = 'not phi.isnull() & hi_max > 1e15 & (vx_rot * {rx} + vy_rot * {ry})>0 & not sim.str.endswith("r")'

new_data = defaultdict(list)
for _, row in tqdm.tqdm(dff.iterrows(), total=len(dff)):
    tbl = tbl_dict[row.sim]
    snap_data = tbl.loc[row.snap-1]
    for c in columns:
        new_data[c].append(snap_data[c])

for c in columns:
    dff[c] = new_data[c]

# convert density
# dff['rho_host'] = dff.rho_host * gadget_dens_units.in_units('amu cm**-3')


# k_B = 1.380649e-23 # J/K
# ## mean constituent mass
# MU_C = 0.58 # amu   ## fully ionized

# # (pynbody.units.Unit('amu cm**-3 J K**-1 K amu**-1').in_units('Pa')) == 1e6
# dff['p_host'] = dff.rho_host * k_B * dff.temp_host / MU_C * (pynbody.units.Unit('amu cm**-3 J K**-1 K amu**-1').in_units('Pa'))


# dff['v'] = np.linalg.norm([dff.vx, dff.vy, dff.vz], axis=0)

# # Pa = N/m2 = kg m/s2 /m2 = kg / (s2 m)
# dff['RPS'] = dff.v**2 * dff.rho_host * (pynbody.units.Unit('km**2 s**-2 amu cm**-3').in_units('Pa'))

for i in [*isophote_target]+[*isophote_target_2]:
    dff[f'dc{i}'] = np.linalg.norm([dff[f'xc{i}'], dff[f'yc{i}']],axis=0)

dff.to_pickle('selected_with_multi_iso_and_morph_and_data.pkl')
# dff[['p_host', 'RPS']].plot()
# (dff.RPS/dff.p_host).plot()

