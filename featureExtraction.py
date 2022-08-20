import csv

from sklearn import datasets
import pandas as pd
import kinematics
import angles
from keypoints2d import JOINTS_2D, Joints, Point2d
import pandas as pd
import numpy as np

# ==============================================================================================================================================================
#                                                       FEATURE EXTRACTION 2D VERSION
# ========================================================================================================================================================

PAIR_ANGLES_JOINT = [('Nose', 'Neck', 'RHip'),
                     ('Nose', 'Neck', 'LHip'),
                     ('Neck', 'RHip', 'RKnee'),
                     ('Neck', 'LHip', 'LKnee'),
                     ('RShoulder', 'RElbow', 'RWrist'),
                     ('LShoulder', 'LElbow', 'LWrist'),
                     ('RHip', 'RKnee', 'RAnkle'),
                     ('LHip', 'LKnee', 'LAnkle'),
                     ('RKnee', 'RHip', 'LKnee')]


# -------------------------------------------------------------------------------------------------
#   keypoints_dict: dictionary <joint_name,array of n point 2d>
#   times         : array of n values representing times for each joint of dict
def extract_temporalspace_features(keypoints_dict, times):

    # for each joint
    info = dict()
    for joint_index, joint in enumerate(JOINTS_2D):

        # calculate displacement, velocity, acceleration for each joint
        measures = __get_kinematics_measures(keypoints_dict, joint, times)

        # for each measure created add joint label
        for measure_name in measures.keys():
            joint_label_name = JOINTS_2D[joint_index]

            # joining all complete values in new dict
            info[joint_label_name + measure_name] = measures[measure_name]

    return info


def __get_kinematics_measures(keypoints_dict, joint, times):
    # measures = {
    #     'DisplacementJ': [],
    #     'DisplacementX': [],
    #     'DisplacementY': [],
    #     'VelocityJ': [],
    #     'VelocityX': [],
    #     'VelocityY': [],
    #     'AccelerationJ': [],
    #     'AccelerationX': [],
    #     'AccelerationY': [],
    #     'TangentAngle': []
    # }
    measures = dict()
    measures['DisplacementJ'] = []
    measures['DisplacementX'] = []
    measures['DisplacementY'] = []
    measures['VelocityJ'] = []
    measures['VelocityX'] = []
    measures['VelocityY'] = []
    measures['AccelerationJ'] = []
    measures['AccelerationX'] = []
    measures['AccelerationY'] = []
    measures['TangentAngle'] = []

    for time_idx in range(len(keypoints_dict[joint]) - 1):
        print('calcolo kinematics tempo: ' + str(time_idx) + ' per joint: ' + joint)
        point_i = keypoints_dict[joint][time_idx]
        point_j = keypoints_dict[joint][time_idx + 1]

        t_i = times[time_idx]
        t_j = times[time_idx + 1]

        displ, d_x, d_y = kinematics.displacement(point_i, point_j)
        vel, v_x, v_y = kinematics.velocity(point_i, point_j, t_i, t_j)
        acc, a_x, a_y = kinematics.acceleration(point_i, point_j, t_i, t_j)

        tangent_angle = kinematics.tangent_angle(d_x, d_y)
        # ---------------------------------------------------
        measures['DisplacementJ'].append(displ)
        measures['DisplacementX'].append(d_x)
        measures['DisplacementY'].append(d_y)

        measures['VelocityJ'].append(vel)
        measures['VelocityX'].append(v_x)
        measures['VelocityY'].append(v_y)

        measures['AccelerationJ'].append(acc)
        measures['AccelerationX'].append(a_x)
        measures['AccelerationY'].append(a_y)

        measures['TangentAngle'].append(tangent_angle)

    return measures


'''
def __get_kinematics_measures(keypoints_dict, joint, times):
    # measures = {
    #     'DisplacementJ': [],
    #     'DisplacementX': [],
    #     'DisplacementY': [],
    #     'VelocityJ': [],
    #     'VelocityX': [],
    #     'VelocityY': [],
    #     'AccelerationJ': [],
    #     'AccelerationX': [],
    #     'AccelerationY': [],
    #     'TangentAngle': []
    # }
    measures = dict()

    for time_idx in range(len(keypoints_dict[joint]) - 1):
        print('calcolo kinematics tempo: ' + str(time_idx) + ' per joint: ' + joint)
        point_i = keypoints_dict[joint][time_idx]
        point_j = keypoints_dict[joint][time_idx + 1]

        t_i = times[time_idx]
        t_j = times[time_idx + 1]

        displ, d_x, d_y = kinematics.displacement(point_i, point_j)
        vel, v_x, v_y = kinematics.velocity(point_i, point_j, t_i, t_j)
        acc, a_x, a_y = kinematics.acceleration(point_i, point_j, t_i, t_j)

        tangent_angle = kinematics.tangent_angle(d_x, d_y)
        # ---------------------------------------------------

        try:
            measures['DisplacementJ'].append(displ)
        except KeyError:
            measures['DisplacementJ'] = []
            measures['DisplacementJ'].append(displ)
        try:
            measures['DisplacementX'].append(d_x)
        except KeyError:
            measures['DisplacementX'] = []
            measures['DisplacementX'].append(d_x)
        try:
            measures['DisplacementY'].append(d_y)
        except KeyError:
            measures['DisplacementY'] = []
            measures['DisplacementY'].append(d_y)
        try:
            measures['VelocityJ'].append(vel)
        except KeyError:
            measures['VelocityJ'] = []
            measures['VelocityJ'].append(vel)
        try:
            measures['VelocityX'].append(v_x)
        except KeyError:
            measures['VelocityX'] = []
            measures['VelocityX'].append(v_x)
        try:
            measures['VelocityY'].append(v_y)
        except KeyError:
            measures['VelocityY'] = []
            measures['VelocityY'].append(v_y)
        try:
            measures['AccelerationJ'].append(acc)
        except KeyError:
            measures['AccelerationJ'] = []
            measures['AccelerationJ'].append(acc)
        try:
            measures['AccelerationX'].append(a_x)
        except KeyError:
            measures['AccelerationX'] = []
            measures['AccelerationX'].append(a_x)
        try:
            measures['AccelerationY'].append(a_y)
        except KeyError:
            measures['AccelerationY'] = []
            measures['AccelerationY'].append(a_y)
        try:
            measures['TangentAngle'].append(tangent_angle)
        except KeyError:
            measures['TangentAngle'] = []
            measures['TangentAngle'].append(tangent_angle)

    return measures
'''


def extract_angles_features(keypoints_dict):
    any_joint = JOINTS_2D[0]
    n_frames = len(keypoints_dict[any_joint]) - 1
    angles_info = dict()
    for time_idx in range(n_frames):
        for (joint_0, joint_1, joint_2) in PAIR_ANGLES_JOINT:

            points_of_joint_0 = keypoints_dict[joint_0][time_idx]
            points_of_joint_1 = keypoints_dict[joint_1][time_idx]
            points_of_joint_2 = keypoints_dict[joint_2][time_idx]

            vector_01 = angles.vector_parallel(points_of_joint_0, points_of_joint_1)
            vector_12 = angles.vector_parallel(points_of_joint_1, points_of_joint_2)

            angle = angles.angle_between(vector_01, vector_12)
            try:
                angles_info['Angle' + '-' + joint_0 + joint_1 + joint_2].append(angle)
            except KeyError:
                angles_info['Angle' + '-' + joint_0 + joint_1 + joint_2] = []
                angles_info['Angle' + '-' + joint_0 + joint_1 + joint_2].append(angle)

    return angles_info


def union_features_data(features_dict_array):
    features = dict()
    for features_dict in features_dict_array:
        for feature_name in features_dict.keys():
            features[feature_name] = features_dict[feature_name]
    return features


def extract_and_store_features(keypointsDict, times, store_directory, videoName):
    keypoints_datas = keypointsDict
    times_data = times
    joints = JOINTS_2D

    for times, joints in zip(times_data, joints):
        print('calcolo features su tempo: ' + str(times) + ' per ' + videoName)
        # all joints for all frames
        temporal_space_features = extract_temporalspace_features(keypoints_datas, times_data)
        angles_features = extract_angles_features(keypoints_datas)

        temporalSpaceDf = pd.DataFrame(temporal_space_features)
        anglesDf = pd.DataFrame(angles_features)

        frames = [temporalSpaceDf, anglesDf]

        result = pd.concat(frames, axis=1)

        # feature complete of stride, union of temporalSpace features and angles features
        # no_sln_features_of_side = union_features_data([temporal_space_features, angles_features])
        # dataFrame = pd.DataFrame.from_dict(no_sln_features_of_side)

        # dataFrame.to_csv(store_directory + typeExtracted + videoName + '.csv', index=False)
        # dataFrame.to_csv(store_directory + videoName + '.csv')
        result.to_csv(store_directory + videoName + '.csv')


def toDict4Extraction(poses, visib):
    dictionary = {}
    tmp = Joints()
    for frame in range(len(poses)):
        i = 0
        for joint in tmp.joints:
            if visib[frame][i]:
                joint.append(Point2d(poses[frame][i][0], poses[frame][i][1]))
            i += 1
    i = 0
    for nameJoint in JOINTS_2D:
        dictionary[nameJoint] = tmp.joints[i]
        i += 1

    return dictionary
