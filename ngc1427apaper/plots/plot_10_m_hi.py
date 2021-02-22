import os
import matplotlib
import matplotlib.pylab as plt
import numpy as np
from simulation.data_handler import DataHandler
from simulation.simdata import get_color_from_name, get_styles_from_peri
import cycler
from ngc1427apaper.helper import savefig
# plt.style.use('./MNRAS.mplstyle')


labels = {'avg_mu_e': r"$\bar{\mu}_{e,r'}$ [mag/arcsec$^2$]",
          'cold_gas': r"M$_g$ ($T<15000$ K) [M$_\odot$]",
          'cold_gas_short': r"M$_g$ [M$_\odot$]",
          'mass_star': r'M$_\star$ [M$_\odot]$',
          'sfr': r'SFR [M$_\odot$/yr]',
          'ssfr': r'sSFR [1/yr]',
          'r': 'r [kpc]',
          'mag_sdss_r': "M$_{r'}$",
          'mass_HI': r"M$_{HI}$ (M$_\odot$)",
          }


# d = DataHandler(cache_file='data_d_orbit_sideon_20191212.pkl').data()
big_df = DataHandler(cache_file='data_d_orbit_sideon_20210219.pkl').data_big()

n = 5
color = plt.cm.copper(np.linspace(0, 1,n))
matplotlib.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

nrows = 2
ncols = 2
fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10, 2.5*nrows), dpi=200)
slicer = slice(None, None, 1)

df = big_df.query('name!="62002"')

for i, ((full_name, sim), ax_s) in enumerate(zip(df.groupby('name', sort=False), ax.flat)):
    # if i==0:continue
    name = full_name[:2]
    if int(name) == 62:
        continue
    print(name)
    for (peri, group) in sim.groupby('pericenter', sort=False):

        k = f'{name}p{peri}'

        ax_s.plot(group.t_period, group.m_hi, label=peri, alpha=0.8)

    ax_s.grid(ls=':')
    ax_s.set_title(name)
    # ax_s.set_ylim(None, 6e-9)
    # ax_s.set_yscale('log')
    ax_s.set_ylabel(labels['mass_HI']);
    # ax_s.set_xlim(-0.25, 0.5)

# ax_s.legend(, ncol=1)
ax[0][0].legend(prop={'size':8}, ncol=1);
ax[-1][0].set_xlabel(r"$\tau$");
ax[-1][1].set_xlabel(r"$\tau$");
# fig.suptitle('M$_{dm}^c$/M$_{dm}$');
# From here: https://stackoverflow.com/a/43578952/1611927
# lgnd = ax.legend(loc='upper center', prop={'size': 5}, ncol=5, scatterpoints=1, fontsize=10)
# for handle in lgnd.legendHandles:
#     handle.set_sizes([5]);

# savefig(fig, 'm_hi', ext=".png", dpi=300)
