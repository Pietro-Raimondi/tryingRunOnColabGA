import numpy as np
import keypoints2d as kp


# formula of angle between two direction vectors
# cos(0) = u*v/||u||*||v||
# output in degrees if not specified, radians otherwise
# ref https://portal.tpu.ru/SHARED/k/KONVAL/Sites/English_sites/G/l_Angle_f.htm#:~:text=The%20angle%20between%20two%20lines%20is%20the%20angle%20between%20direction,given%20by%20the%20following%20formula%3A&text=This%20means%20that%20the%20scalar,.
def angle_between(u, v, type='degree'):
    mod_u = np.linalg.norm(u)
    mod_v = np.linalg.norm(v)
    cos_theta = np.dot(u, v) / (mod_u * mod_v)
    theta = np.arccos(cos_theta)
    if type == 'degree':
        theta = np.degrees(theta)
    return np.round(theta, 2)


# ref : https://www.youmath.it/forum/algebra-lineare/58715-retta-per-due-punti-nello-spazio.html
def vector_parallel(P, Q):
    assert isinstance(P, kp.Point2d)
    assert isinstance(Q, kp.Point2d)
    if isinstance(P, kp.Point2d) and isinstance(Q, kp.Point2d):
        l = Q.x - P.x
        m = Q.y - P.y
        v = [l, m]
        return v
