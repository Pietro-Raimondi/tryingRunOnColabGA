from keypoints2d import JOINTS_2D, Joints
from storingModule import saveData
from copy import deepcopy
from dirManager import checkDir, DATA_RESULTS, NO_PEAKS
import horizontalPeaksElimination


class DataPeaksElimination:

    def __init__(self, personCoordinates, personVisibilities):
        self.coordinates = personCoordinates
        self.visibilities = personVisibilities
        self.threshold = 30

    # function to eliminate peaks in height in a coordinates array
    def peaks_elimination(self, coords):
        height = 0
        for i in range(0, len(coords)):
            if coords[i][2]:
                height = coords[i][1]
                break
        for i in range(0, len(coords)):
            if coords[i][2]:
                a = (coords[i][1] < (height - self.threshold))
                b = (coords[i][1] > (height + self.threshold))
                if a or b:
                    coords[i][0] = 0
                    coords[i][1] = 0
                    coords[i][2] = False
        return coords

    def joint_keypoints_peaks_elimination(self):
        jointsTmp = Joints()

        for i in range(len(self.coordinates)):
            for j in range(len(JOINTS_2D)):
                jointsTmp.joints[j].append([self.coordinates[i][j][0], self.coordinates[i][j][1], self.visibilities[i][j]])

        # deleting peaks in Y
        for j in range(len(JOINTS_2D)):
            jointsTmp.joints[j] = self.peaks_elimination(jointsTmp.joints[j])

        # deleting peaks in X
        jointsTmp.joints[12], jointsTmp.joints[9], jointsTmp.joints[13], jointsTmp.joints[10] = \
            horizontalPeaksElimination.horizontalPeaks(jointsTmp.joints[12], jointsTmp.joints[9], jointsTmp.joints[13], jointsTmp.joints[10])

        for i in range(len(self.coordinates)):
            for j in range(len(JOINTS_2D)):
                self.coordinates[i][j] = [jointsTmp.joints[j][i][0], jointsTmp.joints[j][i][1]]
                self.visibilities[i][j] = jointsTmp.joints[j][i][2]

        return self.coordinates, self.visibilities


# data = videoData class, not a copy
def noPeaksData(data, path2Peaks):
    poses = deepcopy(data.person.pose2d)
    visib = deepcopy(data.person.visibilities)
    data.withoutPeaks.times = deepcopy(data.person.times)
    pelim = DataPeaksElimination(poses, visib)
    data.withoutPeaks.pose2d, data.withoutPeaks.visibilities = pelim.joint_keypoints_peaks_elimination()
    saveData(data, path2Peaks)


def noPeaks(direction, peopleData, poseModel):
    for n in peopleData:
        tmp = peopleData.get(n)
        if tmp.blockName == 'p2s3' and tmp.videoName == 'c3_0330':
            print('here')
        checkDir(poseModel + direction + DATA_RESULTS + tmp.blockName + '/' + tmp.videoName + '/' + NO_PEAKS)
        path2peaks = poseModel + direction + DATA_RESULTS + tmp.blockName + '/' + tmp.videoName + '/' + NO_PEAKS
        noPeaksData(tmp, path2peaks)


def operator(a, b, x, order):
    if order == 'increasing':
        ar = x > a
        br = x < b
        return ar, br
    elif 'descending':
        ar = x < a
        br = x > b
        return ar, br
