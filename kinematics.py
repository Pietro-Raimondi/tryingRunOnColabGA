from keypoints2d import Point2d
import numpy as np


# calculate displacement ------------------------------------------------
# input: position s_0 and s_1 point3s
# output : displacement, displacement_x,displacement_z , displacement_y
def displacement(s_0, s_1):
    delta_x = abs(s_0.x - s_1.x)
    delta_y = abs(s_0.y - s_1.y)
    if isinstance(s_0, Point2d) and isinstance(s_1, Point2d):
        disp = np.sqrt(delta_x ** 2 + delta_y ** 2)
        return np.round(disp, 2), np.round(delta_x, 2), np.round(delta_y, 2)

# -------------------------------------------------------------------------------------------------------

# calculate velocity
# output velocity, velocity x, z, y


def velocity(s_0, s_1, t_0: float, t_1: float):
    if isinstance(s_0, Point2d) and isinstance(s_1, Point2d):
        _, delta_x, delta_y = displacement(s_0, s_1)
        delta_t = t_1 - t_0
        if delta_t == 0:
            if delta_x > 0 or delta_y > 0:
                print('QUI')
        v_x = delta_x / delta_t
        v_y = delta_y / delta_t

        vel = np.sqrt(v_x ** 2 + v_y ** 2)
        return np.round(vel, 2), np.round(v_x, 2), np.round(v_y, 2)

# ---------------------------------------------------------------------------------------------------------


# calculate acceleration
# output acceleration, acceleration x, z, y
def acceleration(s_0, s_1, t_0: float, t_1: float):

    if isinstance(s_0, Point2d) and isinstance(s_1, Point2d):
        _, vx, vy = velocity(s_0, s_1, t_0, t_1)
        delta_t = t_1 - t_0

        ax = vx / delta_t
        ay = vy / delta_t

        acc = np.sqrt(ax ** 2 + ay ** 2)
        return np.round(acc, 2), np.round(ax, 2), np.round(ay, 2)


def tangent_angle(delta_x: float, delta_y: float):
    if delta_x != 0:
        return np.round(np.degrees(np.arctan(delta_y / delta_x)), 2)
    else:
        return np.round(90, 2)
