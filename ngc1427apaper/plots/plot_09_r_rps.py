#! /usr/bin/env python
import matplotlib.pyplot as plt
import pandas as pd
from ngc1427apaper.helper import *
from simulation.simdata import get_tables, SIM_NAME_DICT, get_mach, get_traj, get_cii

from simulation.data_handler import DataHandler
from simulation.units import gadget_dens_units
from pynbody.units import Unit

# table_columns = ['mass_star', 'r_eff3d', 'sfr', 'rho_host']
# mach_columns = ['mach', 'temp_host', 'rho_host', 'v_host']

if __name__ == '__main__':
    sim_label = '69p200'
    sim_name = SIM_NAME_DICT[sim_label]
    tbl_dict = dict()
    # for k,v in SIM_NAME_DICT.items():
    # _t = get_tables(sim_name, True)[table_columns].to_pandas()
    traj_columns = 't,x,y,z,vx,vy,vz,temp_host,rho_host,v_host'.split(',')
    cii_columns = ['cii'] # erg s**-1
    _t = get_traj(sim_name)[traj_columns].to_pandas()
    # _m = get_mach(sim_name)[mach_columns].to_pandas()
    _cii = get_cii(sim_name)[cii_columns].to_pandas()
    # tbl_dict[k] = pd.concat([_t, _m, _cii], axis=1, sort=False)
    df = pd.concat([_t, _cii], axis=1, sort=False)


    # dh = DataHandler(cache_file='data_d_orbit_sideon_20210222.pkl')
    # df = dh.data()['69p100']

    # convert density
    df['rho_host'] = df.rho_host * gadget_dens_units.in_units('amu cm**-3')


    k_B = 1.380649e-23 # J/K
    ## mean constituent mass
    MU_C = 0.58 # amu   ## fully ionized

    # (Unit('amu cm**-3 J K**-1 K amu**-1').in_units('Pa')) == 1e6
    df['p_host'] = df.rho_host * k_B * df.temp_host / MU_C * (Unit('amu cm**-3 J K**-1 K amu**-1').in_units('Pa'))


    # df['v'] = np.linalg.norm([df.vx, df.vy], axis=0)
    df['r'] = np.linalg.norm([df.x, df.y, df.z], axis=0)

    # Pa = N/m2 = kg m/s2 /m2 = kg / (s2 m)
    df['RPS'] = df.v_host**2 * df.rho_host * (Unit('km**2 s**-2 amu cm**-3').in_units('Pa'))

#     dff = get_data('selected_with_multi_iso_and_morph_and_data.pkl')
#     df = df.query('sim=="69p200" & rp==137 & vp==-693 & sign==1')
    fig, ax = plt.subplots()

    # ax.scatter(df_ok.x, df_ok.y, color='b', marker='.')
    # ax.scatter(df_not.x, df_not.y, color='r', marker='.')

    ax.scatter(x=df.r,y=df.RPS, c=df.t, marker='+')

    ax.set_xlabel("r (kpc))")
    # ax.set_ylabel(r"v (km/s)"))
    ax.set_yscale('log')
    ax.set_ylabel(r"RPS ($\rho v^2$) (Pa)")
    ax.grid(ls=':')

    # Savefig
    # file_stem = f"rps"
    # savefig(fig, file_stem, '.png', dpi=150)
    # savefig(fig, file_stem, '.pdf', dpi=150)


    plt.show()