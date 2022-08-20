from copy import deepcopy


class DataVideoTest:
    def __init__(self):
        self.blockName = ''
        self.videoName = ''
        self.pathToFile = ''
        self.pathRawFrames = ''
        self.pathToPose = ''
        self.pathToJson = ''

        self.cap = None
        self.fps = None
        self.flip = False

        self.standard_width = None
        self.standard_height = None
        self.numRawFrames = None
        self.numPoseFrame = None

        self.person = None
        self.withoutPeaks = None
        self.estimated = None
        self.estOnNoPeaks = []
        self.kalmanEstimated = []
        self.kalmanEstOnNoPeaks = []


# class used as tool for preprocessing operations
class Person:
    def __init__(self, joint, visib, time):
        self.pose2d = deepcopy(joint)
        self.visibilities = deepcopy(visib)
        self.times = deepcopy(time)
