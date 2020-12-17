import numpy as np
from .helper import normify
from matplotlib.patches import Arc

def ellipseparams2aperture(params, **kwargs):
    from photutils import EllipticalAperture
    center = params[0:2]
    a, b = params[2:4]
    # a, b = np.max(ab), np.min(ab)
    # a, b = ab
    theta = params[4]
    if a < b:
        a, b = b, a
        theta += np.pi/2
    if theta > 2*np.pi:
        theta = theta % 2*np.pi
    apertures = EllipticalAperture(center, a, b, theta=theta)
    return apertures

def draw_arrows_from_vec(ax, p, v, h, s, color_p='k'):
    ax.set_aspect('equal')
    lim = 1
    # ax.set_xlim(-lim, lim)
    # ax.set_ylim(-lim, lim)
    head_width = 3e-2
    pos = ax.arrow(0.5,0.5, p[0], p[1], length_includes_head=True, head_width=head_width, color=color_p, transform=ax.transAxes)#, ls=':' if v[2] < 0 else '-')
    # ax.arrow(0.5,0.5, v[0], v[1], length_includes_head=True, head_width=head_width, color='g')#, ls=':' if v[2] < 0 else '-')
    hi = ax.arrow(0.5,0.5, h[0], h[1], length_includes_head=True, head_width=head_width, color='r', transform=ax.transAxes)#, ls=':' if v[2] < 0 else '-')
    star = ax.arrow(0.5,0.5, s[0], s[1], length_includes_head=True, head_width=head_width, color='b', transform=ax.transAxes)#, ls=':' if v[2] < 0 else '-')
    # ap_hi, ap_sb
    # return pos, hi, star


def draw_arrows(ax, op, ov, theta_hi, theta_star):
    ax.set_aspect('equal')
    lim = 1
    # ax.set_xlim(-lim, lim)
    # ax.set_ylim(-lim, lim)
    head_width = 3e-2
    fac = 3
    p = -normify(op, fac)
    v = normify(ov, fac)
    h = normify(np.array([np.cos(theta_hi), np.sin(theta_hi)]), fac)
    s = normify(np.array([np.cos(theta_star), np.sin(theta_star)]), fac)
    pos = ax.arrow(0.5,0.5, p[0], p[1], length_includes_head=True, head_width=head_width, color='w', transform=ax.transAxes)#, ls=':' if v[2] < 0 else '-')
    # ax.arrow(0.5,0.5, v[0], v[1], length_includes_head=True, head_width=head_width, color='g')#, ls=':' if v[2] < 0 else '-')
    hi = ax.arrow(0.5,0.5, h[0], h[1], length_includes_head=True, head_width=head_width, color='r', transform=ax.transAxes)#, ls=':' if v[2] < 0 else '-')
    star = ax.arrow(0.5,0.5, s[0], s[1], length_includes_head=True, head_width=head_width, color='b', transform=ax.transAxes)#, ls=':' if v[2] < 0 else '-')
    # ap_hi, ap_sb
    # return pos, hi, star

def _get_angle_three_points(p0, p1=np.array([0,0]), p2=None):
    ''' compute angle (in degrees) for p0p1p2 corner
    Inputs:
        p0,p1,p2 - points in the form of [x,y]
    '''
    if p2 is None:
        p2 = p1 + np.array([1, 0])
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)

    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
    return np.degrees(angle)

def get_angle(p0, p1):
    angle = np.arccos(np.dot(p0, p1)/(np.linalg.norm(p0) * np.linalg.norm(p1)))
    return np.degrees(angle)

def _rotation_transform(theta):
    ''' rotation matrix given theta
    Inputs:
        theta    - theta (in degrees)
    '''
    theta = np.radians(theta)
    A = [[np.math.cos(theta), -np.math.sin(theta)],
         [np.math.sin(theta), np.math.cos(theta)]]
    return np.array(A)

# Adapted from https://stackoverflow.com/a/26417252/1611927
def add_corner_arc(ax, lxy, radius=.7, color=None, text=None, text_radius=.5, text_rotatation=0, text_color='k', **kwargs):
    ''' display an arc for p0p1p2 angle
    Inputs:
        ax     - axis to add arc to
        # line   - MATPLOTLIB line consisting of 3 points of the corner
        radius - radius to add arc
        color  - color of the arc
        text   - text to show on corner
        text_radius     - radius to add text
        text_rotatation - extra rotation for text
        kwargs - other arguments to pass to Arc
    '''

    # lxy = line.get_xydata()
    print(lxy)
    if len(lxy) < 3:
        raise ValueError('at least 3 points in line must be available')

    p0 = lxy[0]
    p1 = lxy[1]
    p2 = lxy[2]

    # width = np.ptp([p0[0], p1[0], p2[0]])
    # height = np.ptp([p0[1], p1[1], p2[1]])
    # st.write(width, height)
    n = 1 # np.array([width, height]) * 1.0
    p0_ = (p0 - p1)
    p1_ = (p1 - p1)
    p2_ = (p2 - p1)

    theta0 = -_get_angle_three_points(p0_, p1_)
    theta1 = -_get_angle_three_points(p2_, p1_)
    # if color is None:
    #     # Uses the color line if color parameter is not passed.
    #     color = line.get_color()
    arc = ax.add_patch(Arc(np.array(p1)+0.5, radius, radius, 0, theta0, theta1, color=color, transform=ax.transAxes, **kwargs))
    if text:
        v = p2_ / np.linalg.norm(p2_)
        if theta0 < 0:
            theta0 = theta0 + 360
        if theta1 < 0:
            theta1 = theta1 + 360
        theta = (theta0 - theta1) / 2 + text_rotatation
        # st.write(theta)
        # if theta > 90:
        #     v*=-1
        # pt = np.dot(_rotation_transform(theta), v[:,None]).T * n * text_radius
        pt = np.dot(_rotation_transform(theta), v).T * text_radius
        # pt = np.array(np.cos(theta), np.sin(theta)) * text_radius
        pt = pt + p1
        pt = pt.squeeze()
        # st.write(theta, v)
        ax.text(pt[0]+0.5, pt[1]+0.5, text,
                horizontalalignment='center', color=text_color,
                verticalalignment='center',transform=ax.transAxes)

    return arc

def get_alpha_beta(orbital_position_rotated, theta_hi, theta_star):
    fac = 1
    p = -normify(orbital_position_rotated, fac)[0:2]
    # v = normify(orbital_velocity_rotated, fac)[0:2]
    h = np.cos(theta_hi), np.sin(theta_hi)
    s = np.cos(theta_star), np.sin(theta_star)
    alpha = get_angle(p, s)
    beta = get_angle(h, s)

    return alpha, beta

