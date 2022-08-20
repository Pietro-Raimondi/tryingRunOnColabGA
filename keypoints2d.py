import math

JOINTS_BODY_25 = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist', 'MidHip', 'RHip',
                  'Rknee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'REye', 'LEye', 'REar', 'LEar', 'LBigToe', 'LSmallToe',
                  'LHeel', 'RBigToe', 'RSmallToe', 'RHeel']

JOINTS_2D = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist', 'RHip', 'RKnee',
             'RAnkle', 'LHip', 'LKnee', 'LAnkle']


class Joints:
    def __init__(self):
        Nose, Neck, RShoulder, RElbow, RWrist, LShoulder, LElbow, LWrist, RHip, Rknee, RAnkle, LHip, LKnee, LAnkle = \
            [], [], [], [], [], [], [], [], [], [], [], [], [], []

        self.joints = [Nose, Neck, RShoulder, RElbow, RWrist, LShoulder, LElbow, LWrist, RHip, Rknee, RAnkle, LHip,
                       LKnee, LAnkle]


# classe che modella il concetto di punto2d
class Point2d:
    def __init__(self, x, y):
        self.__x = x
        self.__y = y

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    def euclidian_distance(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def __str__(self):
        return '(' + str(self.__x) + ', ' + str(self.__y) + ')'

    def __repr__(self):
        return self.__str__()
