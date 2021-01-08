import matplotlib.pyplot as plt
import pandas as pd
from ngc1427apaper.helper import *

if __name__ == '__main__':

    dff = get_data('cache_with_iso_and_hi_valid.pkl')


    df = dff.query('sim=="69p200" & rp==137 & vp==-693 & sign==1')

    df_ok = df.query('solution_found')
    df_not = df.query('not solution_found')

    fig, ax = plt.subplots()

    ax.scatter(df_ok.x, df_ok.y, color='b', marker='.')
    ax.scatter(df_not.x, df_not.y, color='r', marker='.')

    ax.scatter(x=0,y=0,marker='+')
    ax.set_aspect('equal')
    ax.set_xlim(-900, 900)
    ax.set_ylim(-900, 900)
    ax.set_xlabel("x/kpc")
    ax.set_ylabel("y/kpc")
    ax.grid(ls=':')

    # Savefig
    savefig(fig, file_stem = f"good_traj", ext='.pdf')


    fig, ax = plt.subplots()

    ax.scatter(df.t, df.phi, color='b', marker='.', label=r'$\varphi$')
    ax.scatter(df.t, df.theta, color='k', marker='.', label=r'$\theta$')
    ax.set_xlabel("t (Gyr)")
    ax.set_ylabel("angle (deg)")
    ax.grid(ls=':')

    ax.legend()

    savefig(fig, file_stem="angles", ext='.pdf')

    plt.show()