from copy import deepcopy
import keypoints2d as kp
import numpy as np
from pykalman import KalmanFilter


def restructureData(data, caller):
    for n in data:
        postKalman = {}
        tmp = data.get(n)
        preKalman = toDict4Kalman(tmp, caller)
        if caller == 'estimated':
            tmp.kalmanEstimated.append(preKalman)
            tmp.kalmanEstimated.append(postKalman)
        if caller == 'estOnNoPeaks':
            tmp.kalmanEstOnNoPeaks.append(preKalman)
            tmp.kalmanEstOnNoPeaks.append(postKalman)


# originalData, not a copy
def toDict4Kalman(data, caller):
    poses, visib = [], []
    dictionary = {}
    Nose = []
    Neck = []
    RShoulder = []
    RElbow = []
    RWrist = []
    LShoulder = []
    LElbow = []
    LWrist = []
    RHip = []
    RKnee = []
    RAnkle = []
    LHip = []
    LKnee = []
    LAnkle = []
    if caller == 'estimated':
        poses = deepcopy(data.estimated.pose2d)
        visib = deepcopy(data.estimated.visibilities)
    if caller == 'estOnNoPeaks':
        poses = deepcopy(data.estOnNoPeaks[0])
        visib = deepcopy(data.estOnNoPeaks[1])

    for frame in range(len(poses)):
        if visib[frame][0]: Nose.append(kp.Point2d(poses[frame][0][0], poses[frame][0][1]))
        if visib[frame][1]: Neck.append(kp.Point2d(poses[frame][1][0], poses[frame][1][1]))
        if visib[frame][2]: RShoulder.append(kp.Point2d(poses[frame][2][0], poses[frame][2][1]))
        if visib[frame][3]: RElbow.append(kp.Point2d(poses[frame][3][0], poses[frame][3][1]))
        if visib[frame][4]: RWrist.append(kp.Point2d(poses[frame][4][0], poses[frame][4][1]))
        if visib[frame][5]: LShoulder.append(kp.Point2d(poses[frame][5][0], poses[frame][5][1]))
        if visib[frame][6]: LElbow.append(kp.Point2d(poses[frame][6][0], poses[frame][6][1]))
        if visib[frame][7]: LWrist.append(kp.Point2d(poses[frame][7][0], poses[frame][7][1]))
        if visib[frame][8]: RHip.append(kp.Point2d(poses[frame][8][0], poses[frame][8][1]))
        if visib[frame][9]: RKnee.append(kp.Point2d(poses[frame][9][0], poses[frame][9][1]))
        if visib[frame][10]: RAnkle.append(kp.Point2d(poses[frame][10][0], poses[frame][10][1]))
        if visib[frame][11]: LHip.append(kp.Point2d(poses[frame][11][0], poses[frame][11][1]))
        if visib[frame][12]: LKnee.append(kp.Point2d(poses[frame][12][0], poses[frame][12][1]))
        if visib[frame][13]: LAnkle.append(kp.Point2d(poses[frame][13][0], poses[frame][13][1]))

    dictionary['Nose'] = Nose
    dictionary['Neck'] = Neck
    dictionary['RShoulder'] = RShoulder
    dictionary['RElbow'] = RElbow
    dictionary['RWrist'] = RWrist
    dictionary['LShoulder'] = LShoulder
    dictionary['LElbow'] = LElbow
    dictionary['LWrist'] = LWrist
    dictionary['RHip'] = RHip
    dictionary['RKnee'] = RKnee
    dictionary['RAnkle'] = RAnkle
    dictionary['LHip'] = LHip
    dictionary['LKnee'] = LKnee
    dictionary['LAnkle'] = LAnkle
    return dictionary


def kalmanFiltering(data, caller):
    for n in data:
        if caller == 'estimated':
            tmpD = data.get(n).kalmanEstimated[0]
            data.get(n).kalmanEstimated[1] = kalmanSmother(tmpD)
        if caller == 'estOnNoPeaks':
            tmpD = data.get(n).kalmanEstOnNoPeaks[0]
            data.get(n).kalmanEstOnNoPeaks[1] = kalmanSmother(tmpD)


# original_movement_data , it's a dict
#                            ['Nose'] => [(x_0,z_0,y_0)...,(x_n,z_,,y_n)] with n temporal steps
#                            [Neck] => [(x_0,z_0,y_0)...,(x_n,z_,,y_n)]
#                                         .  .  .
def kalmanSmother(original_movement_data):
    x_to_smooth = []
    y_to_smooth = []

    any_joint = kp.JOINTS_2D[0]

    n_frames = len(original_movement_data[any_joint])
    n_joints = len(kp.JOINTS_2D)

    for frame_nro in range(n_frames):
        x_for_joint = []
        y_for_joint = []

        joints = kp.JOINTS_2D
        for joint in joints:
            x_for_joint.append(original_movement_data[joint][frame_nro].x)
            y_for_joint.append(original_movement_data[joint][frame_nro].y)

        x_to_smooth.append(x_for_joint)
        y_to_smooth.append(y_for_joint)
    x_to_smooth = np.asarray(x_to_smooth)
    y_to_smooth = np.asarray(y_to_smooth)

    # -------------------------X SMOOTHING----------------------------------
    kf = KalmanFilter(initial_state_mean=x_to_smooth[0], n_dim_obs=n_joints)
    measurements = x_to_smooth
    kf = kf.em(measurements, n_iter=5)
    (smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)

    for i in range(len(smoothed_state_means)):
        for j in range(len(smoothed_state_means[0])):
            smoothed_state_means[i][j] = np.round(smoothed_state_means[i][j])
    # re-assigning data to return
    x_to_smooth = smoothed_state_means

    # -------------------------y SMOOTHING----------------------------------

    kf = KalmanFilter(initial_state_mean=y_to_smooth[0], n_dim_obs=n_joints)
    measurements = y_to_smooth
    kf = kf.em(measurements, n_iter=5)
    (smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)

    for i in range(len(smoothed_state_means)):
        for j in range(len(smoothed_state_means[0])):
            smoothed_state_means[i][j] = np.round(smoothed_state_means[i][j])
    # re-assigning data to return
    y_to_smooth = smoothed_state_means

    smoothed_movement_data = dict()
    # print(original_movement_data)

    for i, joint in enumerate(joints):
        smoothed_movement_data[joint] = []
        # print(joint)
        for x, y in zip(x_to_smooth[:, i], y_to_smooth[:, i]):
            smoothed_movement_data[joint].append(kp.Point2d(x, y))

    return smoothed_movement_data
