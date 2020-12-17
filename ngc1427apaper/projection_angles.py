import os
import streamlit as st
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R


from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


Vec = namedtuple('Vec', 'x y z')


class Problem:
    def __init__(self, b, C, d, E):
        self.b = b
        self.C = C
        self.d = d
        self.E = E
        p = (b.x * E - d.x * C) / (b.x*d.y-d.x*b.y)
        q = (b.x * d.z - d.x*b.z) / (b.x*d.y-d.x*b.y)
        self.p = p
        self.q = q
        self.a1 = (b.z - b.y*q)**2 + b.x**2 * (1+q**2)
        self.a2 = 2 * ( b.y*b.z*p - p*q*(b.y**2 + b.x**2) + C*(b.y*q - b.z))
        self.a3 = (b.y**2 + b.x**2) * p**2 - 2 * b.y * C * p - b.x**2 + C**2

        self.Delta = self.a2**2 - 4 * self.a1 * self.a3
        if self.Delta < 0:
            raise RuntimeError("No solutions")

    def y_calc(self, z):
        return self.p - self.q * z
        # b = self.b
        # C = self.C
        # d = self.d
        # E = self.E
        # return ((b.x * E - d.x * C) - (b.x*d.z-d.x*b.z))/(b.x*d.y-d.x*b.y) * z

    @property
    def z(self):
        return np.array([( -self.a2 + np.sqrt(self.Delta))/(2*self.a1), (-self.a2 - np.sqrt(self.Delta))/(2*self.a1)])

    @property
    def y(self):
        return self.p - self.q * self.z

    @property
    def x(self):
        # return (self.C - self.y * self.b.y - self.z * self.b.z) / self.b.x
        return (self.C - self.y * self.b.y - self.z * self.b.z) / self.b.x



class Problem2:
    def __init__(self, b, C, d, E):
        self.b = b
        self.C = C
        self.d = d
        self.E = E
        p = (b.x * E - d.x * C) / (b.x*d.y-d.x*b.y)
        # if np.isnan(p): p=0
        q = (b.x * d.z - d.x*b.z) / (b.x*d.y-d.x*b.y)
        # if np.isnan(q): q=0
        m = C - p*b.y
        n = b.z - q*b.y
        # if np.isnan(m): m=0
        # if np.isnan(n): n=0
        self.m = m
        self.n = n
        self.p = p
        self.q = q
        self.a1 = n**2 + b.x**2 * (1+q**2)
        self.a2 = -2 * (m*n + p*q)
        self.a3 = m**2 + b.x**2 * (p**2 -1)

        self.Delta = self.a2**2 - 4 * self.a1 * self.a3
        if self.Delta < 0:
            raise RuntimeError("No solutions")

    def y_calc(self, z):
        return self.p - self.q * z
        # b = self.b
        # C = self.C
        # d = self.d
        # E = self.E
        # return ((b.x * E - d.x * C) - (b.x*d.z-d.x*b.z))/(b.x*d.y-d.x*b.y) * z

    @property
    def z(self):
        return np.array([( -self.a2 + np.sqrt(self.Delta))/(2*self.a1), (-self.a2 - np.sqrt(self.Delta))/(2*self.a1)])
        # more numerically stable: from https://people.csail.mit.edu/bkph/articles/Quadratics.pdf
        # if self.a2 >=0:
        #     return np.array([( -self.a2 - np.sqrt(self.Delta))/(2*self.a1), 2*self.a3/(-self.a2 - np.sqrt(self.Delta))])
        # else:
        #     return np.array([2*self.a3/( -self.a2 + np.sqrt(self.Delta)), (-self.a2 + np.sqrt(self.Delta))/(2*self.a1)])

    @property
    def y(self):
        return self.p - self.q * self.z

    @property
    def x(self):
        # return (self.C - self.y * self.b.y - self.z * self.b.z) / self.b.x
        return (self.C - self.y * self.b.y - self.z * self.b.z) / self.b.x



class Problem3:
    def __init__(self, r, v, rp, vp, sign=+1):
        self.vp = vp
        self.v = v
        j = np.cross(r, v)
        R = sign * np.sqrt(np.linalg.norm(r)**2-rp**2)
        self.R = R
        p = (r[0] * vp - R * v[0]) / j[2]
        q = j[1] / j[2]
        self.p = p
        self.q = q
        # a = vp/v[0]
        # b = -v[1]/v[0]
        # c = -v[2]/v[0]
        a = vp
        b = -v[1]
        c = -v[2]

        alpha = self.a3 = ((a + b*p)**2 + p**2 - 2)*v[0]
        beta = self.a2 = (2*(a + b*p)*(b*q+c) + 2*p*q)*v[0]
        gamma = self.a1 = ((b*q+c)**2 + q**2 + 1)*v[0]

        self.Delta = self.a2**2 - 4 * self.a1 * self.a3
        if self.Delta < 0:
            raise RuntimeError("No solutions")

    @property
    def z(self):
        return np.array([( -self.a2 + np.sqrt(self.Delta))/(2*self.a1), (-self.a2 - np.sqrt(self.Delta))/(2*self.a1)])

    @property
    def y(self):
        return self.p + self.q * self.z

    @property
    def x(self):
        return (self.vp - self.y * self.v[1] - self.z * self.v[2]) / self.v[0]



class Problem4:
    def __init__(self, r, v, rp, vp, sign=+1):
        self.vp = vp
        self.v = v
        j = np.cross(r, v)
        r_norm2 = np.linalg.norm(r)**2
        v_norm2 = np.linalg.norm(v)**2
        j_norm = np.linalg.norm(j)
        R = sign * np.sqrt(r_norm2-rp**2)
        self.R = R

        dot = np.dot(r,v)
        b = (R*v_norm2 - vp * dot)/(r_norm2*v_norm2 - dot**2)
        a = (vp - b * dot)/v_norm2
        self.a = a
        self.b = b
        num = 1-a**2*v_norm2-b**2*r_norm2-2*a*b*dot
        self.num = num
        if num < 0:
            # raise RuntimeError("No solutions")
            self.x = self.y = self.z = np.array([np.nan, np.nan])
        else:
            c1 = np.sqrt(num)/j_norm
            c2 = -np.sqrt(num)/j_norm
            sol1 = a*v+b*r+c1*j
            sol2 = a*v+b*r+c2*j
            self.x = np.array([sol1[0], sol2[0]])
            self.y = np.array([sol1[1], sol2[1]])
            self.z = np.array([sol1[2], sol2[2]])

    def _repr__(self):
        return f'{self.x}, {self.y}, {self.z}'

    def __str__(self):
        return f'{self.x}, {self.y}, {self.z}'

def solve_problem_for_trajectory(tbl_traj, rp, vp, sign_of_r):
    from collections import defaultdict
    import pandas as pd

    d = defaultdict(list)

    for t in tbl_traj:
        r_arr = sign_of_r * np.array([t['x'], t['y'], t['z']])
        v_arr = np.array([t['vx'], t['vy'], t['vz']])
        # st.write(r_arr)
        # st.write(v_arr)
        try:
            p = Problem4(r=r_arr, v=v_arr, rp=rp, vp=vp, sign=sign_of_r)
            x, y, z = p.x, p.y, p.z
        except RuntimeError:
            x = y = z = np.array([np.nan, np.nan])

        d['x1'].append(x[0])
        d['y1'].append(y[0])
        d['z1'].append(z[0])
        d['x2'].append(x[1])
        d['y2'].append(y[1])
        d['z2'].append(z[1])

    df = pd.DataFrame(d)

    # df = pd.concat([tbl_traj.to_pandas(), pd.DataFrame(d)], axis=1)

    df['phi1'] = np.arctan2(df.y1, df.x1)*180/np.pi
    df['phi2'] = np.arctan2(df.y2, df.x2)*180/np.pi
    df['theta1'] = np.arccos(df.z1)*180/np.pi
    df['theta2'] = np.arccos(df.z2)*180/np.pi
    return df

def draw_solutions(sol, i):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(sol[f'theta1'], 'r', label=f'theta1')
    ax.plot(sol[f'phi1'], 'r:', label=f'phi1')
    ax.plot(sol[f'theta2'], 'k', label=f'theta2')
    ax.plot(sol[f'phi2'], 'k:', label=f'phi2')
    ax.axvline(i)
    ax.legend()
    ax.set_xlim(0, len(sol))
    ax.set_xlabel('snap')
    ax.set_ylabel('angle (deg)')
    ax.grid()
    return fig



def get_targets(sol, tbl_traj, which_solution):

    r_arr = np.array([tbl_traj['x'], tbl_traj['y'], tbl_traj['z']])
    v_arr = np.array([tbl_traj['vx'], tbl_traj['vy'], tbl_traj['vz']])
    phi = sol[f'phi{which_solution}'].to_numpy()*np.pi/180
    theta = sol[f'theta{which_solution}'].to_numpy()*np.pi/180

    r1 = R.from_rotvec(phi[:,np.newaxis] * np.array([0, 0, 1]))

    # r_prime = r1.apply(r_arr.T)
    # v_prime = r1.apply(v_arr.T)

    # x_prime = r1.apply(np.array([1, 0, 0]))
    y_prime = r1.apply(np.array([0, 1, 0]))
    # z_prime = r1.apply(np.array([0, 0, 1]))

    r2 = R.from_rotvec(theta[:, np.newaxis] * y_prime)

    # x_second = r2.apply(x_prime)
    # y_second = r2.apply(y_prime)
    # z_second = r2.apply(z_prime)

    # st.write(np.array([x_second, y_second, z_second]).transpose(1,0,2))
    r_inv = (r2 * r1).inv()
    # r_inv = R.from_matrix(np.array([x_second, y_second, z_second]).transpose(1,0,2))
    orbital_position_rotated = r_inv.apply(r_arr.T)
    orbital_velocity_rotated = r_inv.apply(v_arr.T)

    # orbital_position_rotated = r2.apply(r_prime)
    # orbital_velocity_rotated = r2.apply(v_prime)

    # st.write("$|r'|$:", np.linalg.norm(orbital_position_rotated), "$|v'|$:", np.linalg.norm(orbital_velocity_rotated))

    # r_xy = np.linalg.norm(orbital_position_rotated[0:2])
    r_xy = np.sqrt(orbital_position_rotated[:,0]**2 + orbital_position_rotated[:,1]**2)
    los_velocity = -orbital_velocity_rotated[:,2]
    return r_xy, los_velocity




    x_prime = r1.apply(np.array([1, 0, 0]))
    y_prime = r1.apply(np.array([0, 1, 0]))
    z_prime = r1.apply(np.array([0, 0, 1]))
    # st.write('z_prime', z_prime)
    r2 = R.from_rotvec(theta * y_prime)


    x_second = r2.apply(x_prime)
    y_second = r2.apply(y_prime)
    z_second = r2.apply(z_prime)

    # orbital_position_rotated = r2.apply(r_prime)
    # orbital_velocity_rotated = r2.apply(v_prime)

    r_inv = R.from_matrix(np.array([x_second, y_second, z_second]))
    st.write(r_inv.as_matrix())
    orbital_position_rotated = r_inv.apply(r_arr)
    orbital_velocity_rotated = r_inv.apply(v_arr)







def draw_targets(r_xy, los_velocity, which_snap, rp, vp):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(r_xy, 'r', label='r_xy')
    ax.plot(los_velocity, 'r:', label='vlos')
    ax.axvline(which_snap-1)
    # ax.axhline(rp, ls="-")
    # ax.axhline(vp, ls=":")
    ax.legend()
    ax.set_xlim(0, len(r_xy))
    ax.set_xlabel('snap')
    # ax.set_ylabel('angle (deg)')
    ax.grid()
    return fig

def draw_delta_targets(r_xy, los_velocity, which_snap, rp, vp):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(r_xy-rp, 'r', label='r_xy')
    ax.plot(los_velocity-vp, 'r:', label='vlos')
    ax.axvline(which_snap-1)
    ax.legend()
    ax.set_xlim(0, len(r_xy))
    ax.set_xlabel('snap')
    # ax.set_ylabel('angle (deg)')
    ax.grid()
    return fig


from simulation.simdata import SIM_NAME_DICT, SPECIAL_DICT, get_traj

SIM_DICT = {**SIM_NAME_DICT, **SPECIAL_DICT}

if __name__ == '__main__':

    use_traj = st.sidebar.checkbox('Use trajectory', False)
    _step = 1
    sign_of_r = st.sidebar.radio(r'Sign of r', [+1, -1])
    w = st.sidebar.radio('which solution', [1,2])
    if use_traj:
        sim_label = st.sidebar.selectbox(label='Which sim',
                                         # options=list(map(os.path.basename, snaps)),
                                         options=list(SIM_DICT.keys()),
                                         index=16,
                                         format_func=os.path.basename,
                                         )

        sim_name = SIM_DICT[sim_label]

        which_snap = st.sidebar.number_input('snap', min_value=1, max_value=563, value=89, step=1)

        rp = st.sidebar.slider('rp', min_value=0, max_value=500, value=137, step=_step)
        vp = st.sidebar.slider('vp', min_value=-800, max_value=1000, value=-693, step=_step)
        tbl_traj = get_traj(sim_name)

        # st.write(Vec(*tuple(np.array(cur_pos))))
        # st.write(Vec(*tuple(-np.array(cur_vel))))

        sol = solve_problem_for_trajectory(tbl_traj, rp=rp, vp=vp, sign_of_r=sign_of_r)
        fig1 = draw_solutions(sol, which_snap-1)
        st.write(fig1)

        r_proj, vlos = get_targets(sol, tbl_traj, w)
        fig2 = draw_targets(r_proj, vlos, which_snap, rp, vp)
        st.write(fig2)
        t = tbl_traj[which_snap-1]
        r_arr = np.array([t['x'], t['y'], t['z']])
        v_arr = np.array([t['vx'], t['vy'], t['vz']])
    else:
        # _step = 5
        # rx = st.sidebar.slider('rx', min_value=0, max_value=500, value=145, step=_step)
        # ry = st.sidebar.slider('ry', min_value=0, max_value=500, value=50, step=_step)
        # rz = st.sidebar.slider('rz', min_value=0, max_value=500, value=95, step=_step)

        # 410
        # r_text = st.text_input('r', value='115.48899999999999 252.43 -1.54493')
        # v_text = st.text_input('v', value='-102.12 1124.13 -5.92606')
        # r_text = st.text_input('r', value='-171.09599999999998 -399.81800000000004 1.65335')
        # v_text = st.text_input('v', value='634.484 577.6080000000001 3.9402699999999995')

        # 361
        # r_text = st.text_input('r', value='-98.3027 -328.306 2.17143')
        # v_text = st.text_input('v', value='687.994 729.436 4.61808')
        r_text = st.text_input('r', value='100 0 0 ')
        v_text = st.text_input('v', value='100 100 0')
        rp = st.sidebar.slider('rp', min_value=0, max_value=500, value=60, step=_step)
        vp = st.sidebar.slider('vp', min_value=0, max_value=1000, value=80, step=_step)

    # vx = st.sidebar.slider('vx', min_value=0, max_value=500, value=140, step=_step)
    # vy = st.sidebar.slider('vy', min_value=0, max_value=500, value=50, step=_step)
    # vz = st.sidebar.slider('vz', min_value=0, max_value=500, value=95, step=_step)

    # r = Vec(rx, ry, rz)
    # v = Vec(vx, vy, vz)

        r_arr = np.array(tuple(map(float, r_text.split())))
        v_arr = np.array(tuple(map(float, v_text.split())))
    # r = Vec(*(-1*np.array(tuple(map(float, r_text.split())))))
    # v = Vec(*(+1*np.array(tuple(map(float, v_text.split())))))
    st.write('$r$:', *r_arr)
    st.write('$v$:', *v_arr)

    r_modulus  =  np.linalg.norm(r_arr)
    v_modulus  =  np.linalg.norm(v_arr)
    r_versor = r_arr/r_modulus
    v_versor = v_arr/v_modulus
    st.write('$|r|$:', r_modulus, '$|v|$:', v_modulus)
    st.write('$r_{xy}$:', rp, r'$v_\parallel$:', vp)

    # TODO projection can't be longer that actual vectors r and v

    # # C = r_modulus * np.cos(np.arcsin(rp / r_modulus))
    # C = np.sqrt(r_modulus**2 - rp**2)
    # E = vp
    # p = Problem2(r, C, v, E)
    # p.E, p.C, p.b, p.d, p.m, p.n

    p = Problem4(r=r_arr, v=v_arr, rp=rp, vp=vp, sign=sign_of_r)

    st.write('x:', *p.x)
    st.write('y:', *p.y)
    st.write('z:', *p.z)
    # st.write(f'{p.a1:.1f} $z^2$ {p.a2:+.1f} $z$ {p.a3:+.1f} = 0. $\sqrt{{\Delta}}$ = {np.sqrt(p.Delta):.1f}')
    st.write(f'R={p.R}')
    st.write(f'a={p.a}')
    st.write(f'b={p.b}')
    st.write(f'c{w}={getattr(p, f"c{w}")}')
    st.write(f'num={p.num}')

    phi_tuple = np.array([np.arctan2(p.y[0], p.x[0]), np.arctan2(p.y[1], p.x[1])])
    theta_tuple = np.array([np.arccos(p.z[0]), np.arccos(p.z[1])])
    st.write(r"$\varphi$", tuple(phi_tuple*180/np.pi))
    st.write(r"$\theta$",  tuple(theta_tuple*180/np.pi))


    st.write('Verify\n')

    x1 = np.array([p.x[0], p.y[0], p.z[0]])
    x2 = np.array([p.x[1], p.y[1], p.z[1]])

    zero1 = np.dot(x1, r_arr) - p.R
    zero2 = np.dot(x2, v_arr) - vp

    st.write(np.sqrt(p.x[0]**2+p.y[0]**2+p.z[0]**2), np.sqrt(p.x[1]**2+p.y[1]**2+p.z[1]**2))
    st.write(zero1, zero2)

    st.write('Verify vectors')
    phi = phi_tuple[w-1]
    theta = theta_tuple[w-1]
    # oz1 = phi
    # ox = -theta
    # r1 = R.from_euler('ZY', [oz1, ox], degrees=False)
    # r1 = R.from_euler('Z', phi, degrees=False)
    # r2 = R.from_euler('Y', theta, degrees=False)
    # r1 = R.from_euler('Z', phi-np.pi/2, degrees=False)
    # r2 = R.from_euler('X', -theta, degrees=False)
    # r1 = R.from_rotvec(phi-np.pi/2 * np.array([0, 0, 1]))
    # r2 = R.from_rotvec(-theta * np.array([1, 0, 0]))

    r1 = R.from_rotvec(phi * np.array([0, 0, 1]))

    r_prime = r1.apply(r_arr)
    v_prime = r1.apply(v_arr)

    x_prime = r1.apply(np.array([1, 0, 0]))
    y_prime = r1.apply(np.array([0, 1, 0]))
    z_prime = r1.apply(np.array([0, 0, 1]))
    # st.write('z_prime', z_prime)
    r2 = R.from_rotvec(theta * y_prime)


    x_second = r2.apply(x_prime)
    y_second = r2.apply(y_prime)
    z_second = r2.apply(z_prime)

    # orbital_position_rotated = r2.apply(r_prime)
    # orbital_velocity_rotated = r2.apply(v_prime)

    r_inv = R.from_matrix(np.array([x_second, y_second, z_second]))
    st.write(r_inv.as_matrix())
    orbital_position_rotated = r_inv.apply(r_arr)
    orbital_velocity_rotated = r_inv.apply(v_arr)


    # oz = st.sidebar.slider('Rotate z', min_value=-180, max_value=180, value=0, step=2)
    # r2 = R.from_euler('z', oz, degrees=False)
    # rotation = r2 * r1
    # # st.write(rotation.as_euler('ZXY', degrees=True))

    # # rotation=r1
    # rotation_matrix = rotation.as_matrix()
    # st.write(rotation_matrix)

    # orbital_position_rotated = rotation.apply(np.array(r))
    # orbital_velocity_rotated = rotation.apply(np.array(v))

    st.write("$r''$:", *orbital_position_rotated)
    st.write("$v''$:", *orbital_velocity_rotated)

    # st.write("$|r'|$:", np.linalg.norm(orbital_position_rotated), "$|v'|$:", np.linalg.norm(orbital_velocity_rotated))

    # r_xy = np.linalg.norm(orbital_position_rotated[0:2])
    r_xy = np.sqrt(orbital_position_rotated[0]**2 + orbital_position_rotated[1]**2)
    los_velocity = -orbital_velocity_rotated[2]
    vz_target = orbital_velocity_rotated[2]

    st.markdown(
    f"""
    $|r_{{xy}}^{{jf}}|$ = {rp} kpc, $v_{{z}}^{{jf}}$ = {vp} km/s\n
    $|r_{{xy}}|$ = {r_xy:.0f} kpc, $v_{{z}}$ = {vz_target:.0f} km/s
    """
    )
    # ------
    show_fig = st.sidebar.checkbox('show fig', value=False)
    if show_fig:
        fig = plt.figure(figsize=(12,12), dpi=300)
        ax = fig.add_subplot(111, projection='3d')
        # ax.set_aspect('equal')

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)

        x = 1 * np.outer(np.cos(u), np.sin(v))
        y = 1 * np.outer(np.sin(u), np.sin(v))
        z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))

        ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b',
            linewidth=0,
            alpha=0.2)

        ul = 0.5
        x = Arrow3D([0, ul], [0, 0], [0, 0], mutation_scale=20,
                    lw=3, arrowstyle="-|>", color="r")
        y = Arrow3D([0, 0], [0, ul], [0, 0], mutation_scale=20,
                    lw=3, arrowstyle="-|>", color="g")

        z = Arrow3D([0, 0], [0, 0], [0, ul], mutation_scale=20,
                    lw=3, arrowstyle="-|>", color="b")

        # a = Arrow3D([0, p.x[0]], [0, p.y[0]], [0, p.z[0]], mutation_scale=20,
        #             lw=3, arrowstyle="-|>", color="k", ls='--')
        # b = Arrow3D([0, p.x[1]], [0, p.y[1]], [0, p.z[1]], mutation_scale=20,
        #             lw=3, arrowstyle="-|>", color="c", ls='--')


        _r = Arrow3D([0, r_versor[0]], [0, r_versor[1]], [0, r_versor[2]], mutation_scale=20,
                    lw=2, arrowstyle="-|>", color="k", ls='--')
        _v = Arrow3D([0, v_versor[0]], [0, v_versor[1]], [0, v_versor[2]], mutation_scale=20,
                    lw=2, arrowstyle="-|>", color="c", ls='--')

        # _rs = Arrow3D([0, r_versor[0]], [0, r_versor[1]], [0, r_versor[2]], mutation_scale=20,
        #             lw=2, arrowstyle="-|>", color="k", ls='--')
        # _vs = Arrow3D([0, v_versor[0]], [0, v_versor[1]], [0, v_versor[2]], mutation_scale=20,
        #             lw=2, arrowstyle="-|>", color="c", ls='--')
        # _y = Arrow3D([0, y_prime[0]], [0, y_prime[1]], [0, y_prime[2]], mutation_scale=20,
        #             lw=2, arrowstyle="-|>", color="y", ls='--')
        _xs = Arrow3D([0, x_second[0]], [0, x_second[1]], [0, x_second[2]], mutation_scale=20,
            lw=2, arrowstyle="-|>", color="r", ls='--')

        _ys = Arrow3D([0, y_second[0]], [0, y_second[1]], [0, y_second[2]], mutation_scale=20,
            lw=2, arrowstyle="-|>", color="g", ls='--')

        _zs = Arrow3D([0, z_second[0]], [0, z_second[1]], [0, z_second[2]], mutation_scale=20,
            lw=2, arrowstyle="-|>", color="b", ls='--')
        for _a in (x, y, z, _r, _v, _xs, _ys, _zs):
            ax.add_artist(_a)
        # ax.add_artist(b)

        ax.scatter(p.x[0], p.y[0], p.z[0], s=200)
        ax.scatter(p.x[1], p.y[1], p.z[1], s=200)
        # ax.scatter(0,0,0,marker='+')

        theta_view = st.sidebar.slider('theta_view', min_value=0.0, max_value=180.0, value=float(theta*180/np.pi))
        elev = 90-theta_view
        azim = st.sidebar.slider('Phi', min_value=-180.0, max_value=180.0, value=float(phi*180/np.pi))

        # adapted from here:
        # https://stackoverflow.com/a/32427177
        rot = 0.0 / 180 * np.pi
        longitude = 0
        a = np.array([-np.sin(longitude / 180 * np.pi), 0, np.cos(longitude / 180 * np.pi)])
        b = np.array([0, 1, 0])
        b = b * np.cos(rot) + np.cross(a, b) * np.sin(rot) + a * np.dot(a, b) * (1 - np.cos(rot))
        ax.plot(np.sin(u),np.cos(u),0,color='k', linestyle = 'dashed')
        vert_front = np.linspace(np.pi / 2, 3 * np.pi / 2, 100)
        ax.plot(a[0] * np.sin(u) + b[0] * np.cos(u), b[1] * np.cos(u), a[2] * np.sin(u) + b[2] * np.cos(u),color='k', linestyle = 'dashed')
        ax.plot(a[0] * np.sin(vert_front) + b[0] * np.cos(vert_front), b[1] * np.cos(vert_front), a[2] * np.sin(vert_front) + b[2] * np.cos(vert_front),color='k')

        ax.view_init(elev=elev, azim=azim)

        st.write(fig)
