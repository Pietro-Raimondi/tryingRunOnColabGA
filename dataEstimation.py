import numpy as np
from storingModule import saveData
from copy import deepcopy
from dirManager import checkDir, DATA_RESULTS, ESTIMATED


def keypoints_estimation(coordinates, visibilities):
    # building arrays for interpolations
    # array to put coordinates (all estimations are made singularly for every keypoint)
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
    # storing coordinates by appending them in correspondent arrays
    for i in range(len(coordinates)):
        Nose.append(coordinates[i][0])
        Neck.append(coordinates[i][1])
        RShoulder.append(coordinates[i][2])
        RElbow.append(coordinates[i][3])
        RWrist.append(coordinates[i][4])
        LShoulder.append(coordinates[i][5])
        LElbow.append(coordinates[i][6])
        LWrist.append(coordinates[i][7])
        RHip.append(coordinates[i][8])
        RKnee.append(coordinates[i][9])
        RAnkle.append(coordinates[i][10])
        LHip.append(coordinates[i][11])
        LKnee.append(coordinates[i][12])
        LAnkle.append(coordinates[i][13])
    # estimation of the undetected values
    Nose = estimation(Nose)
    Neck = estimation(Neck)
    RShoulder = estimation(RShoulder)
    RElbow = estimation(RElbow)
    RWrist = estimation(RWrist)
    LShoulder = estimation(LShoulder)
    LElbow = estimation(LElbow)
    LWrist = estimation(LWrist)
    RHip = estimation(RHip)
    RKnee = estimation(RKnee)
    RAnkle = estimation(RAnkle)
    LHip = estimation(LHip)
    LKnee = estimation(LKnee)
    LAnkle = estimation(LAnkle)
    # update person coordinates and visibilities after estimation of undetected values
    for i in range(len(coordinates)):
        coordinates[i][0] = Nose[i]
        visibilities[i][0] = True
        coordinates[i][1] = Neck[i]
        visibilities[i][1] = True
        coordinates[i][2] = RShoulder[i]
        visibilities[i][2] = True
        coordinates[i][3] = RElbow[i]
        visibilities[i][3] = True
        coordinates[i][4] = RWrist[i]
        visibilities[i][4] = True
        coordinates[i][5] = LShoulder[i]
        visibilities[i][5] = True
        coordinates[i][6] = LElbow[i]
        visibilities[i][6] = True
        coordinates[i][7] = LWrist[i]
        visibilities[i][7] = True
        coordinates[i][8] = RHip[i]
        visibilities[i][8] = True
        coordinates[i][9] = RKnee[i]
        visibilities[i][9] = True
        coordinates[i][10] = RAnkle[i]
        visibilities[i][10] = True
        coordinates[i][11] = LHip[i]
        visibilities[i][11] = True
        coordinates[i][12] = LKnee[i]
        visibilities[i][12] = True
        coordinates[i][13] = LAnkle[i]
        visibilities[i][13] = True
    return coordinates, visibilities


# function to estimate the undetected values in a coordinates array
def estimation(coordinates):
    # calculate indexes of undetected values
    index = calcIndex(coordinates)
    # if in the array weren't detected null values, there's no need to do the estimation
    if len(index) > 0:
        # saving the first index
        before_index = index[0]
        # cycle for every index
        for i in range(len(index)):
            # if required to avoid to get index out of bound exception
            if i == len(index) - 1:
                # saving the index over the 'gap' between the two correct values
                after_index = index[i]
                if after_index == len(coordinates) - 1:
                    # estimating every point in this gap, by using linspace method
                    estimated_values_x = np.linspace(coordinates[before_index - 1][0], coordinates[before_index - 1][0],
                                                     after_index - before_index + 2, endpoint=False, dtype=int)
                    estimated_values_y = np.linspace(coordinates[before_index - 1][1], coordinates[before_index - 1][1],
                                                     after_index - before_index + 2, endpoint=False, dtype=int)
                else:
                    # estimating every point in this gap, by using linspace method
                    estimated_values_x = np.linspace(coordinates[before_index - 1][0], coordinates[after_index + 1][0],
                                                     after_index - before_index + 2, endpoint=False, dtype=int)
                    estimated_values_y = np.linspace(coordinates[before_index - 1][1], coordinates[after_index + 1][1],
                                                     after_index - before_index + 2, endpoint=False, dtype=int)
                # cycle to fill every missing value
                for j in range(before_index, after_index + 2):
                    # if made for skipping first linspace value (because the first value correspond to the value given)
                    if j != before_index:
                        # storing values
                        coordinates[j - 1][0] = estimated_values_x[j - before_index]
                        coordinates[j - 1][1] = estimated_values_y[j - before_index]
            # if between this index and the following one there is a distance greater than one,
            # it means there are some values not identified, and this few lines are going to fill them
            elif index[i + 1] - index[i] != 1:
                # saving the index over the 'gap' between the two correct values
                after_index = index[i]
                if before_index == 0:
                    # estimating every point in this gap, by using linspace method
                    estimated_values_x = np.linspace(coordinates[after_index + 1][0],
                                                     coordinates[after_index + 1][0],
                                                     after_index - before_index + 2, endpoint=False, dtype=int)
                    estimated_values_y = np.linspace(coordinates[after_index + 1][1],
                                                     coordinates[after_index + 1][1],
                                                     after_index - before_index + 2, endpoint=False, dtype=int)
                else:
                    # estimating every point in this gap, by using linspace method
                    estimated_values_x = np.linspace(coordinates[before_index - 1][0],
                                                     coordinates[after_index + 1][0],
                                                     after_index - before_index + 2, endpoint=False, dtype=int)
                    estimated_values_y = np.linspace(coordinates[before_index - 1][1],
                                                     coordinates[after_index + 1][1],
                                                     after_index - before_index + 2, endpoint=False, dtype=int)
                # cycle to fill every missing value
                for j in range(before_index, after_index + 2):
                    # if made for skipping first linspace value (because the first value correspond to the value given)
                    if j != before_index:
                        # storing values
                        coordinates[j - 1][0] = estimated_values_x[j - before_index]
                        coordinates[j - 1][1] = estimated_values_y[j - before_index]
                # next step
                before_index = index[i + 1]
    return coordinates


# function to calculate the index of undetected values in a coordinates array
def calcIndex(coordinates):
    index = []
    for i in range(len(coordinates)):
        if coordinates[i][0] == 0:
            index.append(i)
    return index


# data = DataVideoClass, not a copy
def estimatedData(data, path2estimation):
    pose = deepcopy(data.person.pose2d)
    visib = deepcopy(data.person.visibilities)
    data.estimated.times = deepcopy(data.person.times)
    data.estimated.pose2d, data.estimated.visibilities = keypoints_estimation(pose, visib)
    saveData(data, path2estimation)


def estimOnNoPeaks(poses, visibs):
    pose = deepcopy(poses)
    visib = deepcopy(visibs)
    pose, visib = keypoints_estimation(pose, visib)
    return pose, visib


def estimatingData(direction, peopleData, poseModel):
    for n in peopleData:
        tmp = peopleData.get(n)
        checkDir(poseModel + direction + DATA_RESULTS+tmp.blockName+'/'+tmp.videoName+'/'+ESTIMATED)
        path2estimation = poseModel + direction + DATA_RESULTS+tmp.blockName+'/'+tmp.videoName+'/'+ESTIMATED
        estimatedData(tmp, path2estimation)
