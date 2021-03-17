#! /usr/bin/env python
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import cycler
import pandas as pd
from ngc1427apaper.helper import savefig
from simulation.simdata import get_tables, SIM_NAME_DICT, get_mach, get_traj, get_cii

from simulation.data_handler import DataHandler
from simulation.units import gadget_dens_units
from pynbody.units import Unit

# table_columns = ['mass_star', 'r_eff3d', 'sfr', 'rho_host']
# mach_columns = ['mach', 'temp_host', 'rho_host', 'v_host']

if __name__ == '__main__':
    # n = 5
    # color = plt.cm.copper(np.linspace(0, 1,n))
    # matplotlib.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)
    fig, ax = plt.subplots()
    dh = DataHandler(cache_file='data_d_orbit_sideon_20210222.pkl')
    dff = dh.data_big()
    sim_name = 69002
    dfq = dff.query(f'name == "{sim_name}" & pericenter!=100 & pericenter!=200')
    groups = dfq.groupby('pericenter')
    markers = ('+','*', '.')
    for i, (peri, df) in enumerate(groups):
        # if peri in [100, 200]: continue
        # print(df.sim_label.iloc[0])
        # peri = df.pericenter.iloc[0]
        print(peri)
        # df = df.iloc[slice(0, len(df), 5)]
        # ax.plot(x=df.r,y=df.RPS, marker='+')
        # ax.plot(df.r, df.RPS, label=peri)
        mappable = ax.scatter(df.t_period, df.RPS, c=df.r, label=peri, marker=markers[i], alpha=0.8)
        if i==0:
            good_mappable = mappable
    ax.legend()

#     dff = get_data('selected_with_multi_iso_and_morph_and_data.pkl')
#     df = df.query('sim=="69p200" & rp==137 & vp==-693 & sign==1')

    # ax.scatter(df_ok.x, df_ok.y, color='b', marker='.')
    # ax.scatter(df_not.x, df_not.y, color='r', marker='.')
    # ax.set_title(sim_name)

    ax.set_xlabel(r"$\tau$")
    # ax.set_ylabel(r"v (km/s)"))
    ax.set_yscale('log')
    ax.set_ylabel(r"RPS ($\rho v^2$) (Pa)")
    ax.grid(ls=':')
    cbar = fig.colorbar(good_mappable)
    cbar.ax.set_ylabel("r (kpc)")
    # Savefig
    file_stem = f"rps"
    savefig(fig, file_stem, '.png', dpi=300)
    # savefig(fig, file_stem, '.pdf', dpi=150)

    plt.show()
