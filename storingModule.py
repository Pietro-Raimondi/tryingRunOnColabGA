from dirManager import unpicklingData, checkDir, POST_JSON_READING, DATA_RESULTS, ORIGINAL
from keypoints2d import Joints, JOINTS_2D
from extractionAndPose import analyzeModel


def write(directory, body_type, positions, data):
    file = open(directory + "/" + body_type + ".txt", "w")
    file.write("FileType: TXT\n")
    file.write("NumberPoints: " + str(len(positions)) + "\n")
    file.write("VideoWidth: " + str(data.standard_width) + "\n")
    file.write("VideoHeight: " + str(data.standard_height) + "\n")
    file.write("SamplingFrequency(Hz): " + str(data.fps) + "\n")
    time = []
    x_coordinates = []
    y_coordinates = []
    for i in range(len(positions)):
        t = positions[i][0]
        x = positions[i][1]
        y = positions[i][2]
        z = 0.0
        pressure = 1.0
        azimuth = 0.
        elevation = 0.
        tangent = 0.
        file.write('%f' % t + " " + str(x) + " " + str(y) + " " + str(z) + " " + str(pressure) + " " + str(
            azimuth) + " " + str(elevation) + " " + str(tangent) + "\n")
        time.append(t)
        x_coordinates.append(x)
        y_coordinates.append(y)
    file.close()


def saveData(data, data_dir):
    from inspect import stack
    # index = data_dir.find('/')
    # model = data_dir[:index]

    poses, visib, times = None, None, None
    callerMethod = stack()[1].function
    if callerMethod == 'saveOriginalData':
        poses = data.person.pose2d
        visib = data.person.visibilities
        times = data.person.times

    if callerMethod == 'noPeaksData':
        poses = data.withoutPeaks.pose2d
        visib = data.withoutPeaks.visibilities
        times = data.person.times

    if callerMethod == 'estimatedData':
        poses = data.estimated.pose2d
        visib = data.estimated.visibilities
        times = data.person.times

    tmp = Joints()
    for frame in range(len(poses)):
        i = 0
        for joint in tmp.joints:
            if visib[frame][i]:
                joint.append((times[frame], poses[frame][i][0], poses[frame][i][1]))
            i += 1
    i = 0
    for nameJoint in JOINTS_2D:
        write(data_dir, nameJoint, tmp.joints[i], data)
        i += 1
    print("keypoints stored in .txt files")


def saveOriginalData(direction, pm='BODY_25'):
    poseModel = analyzeModel(pm)
    #   index = data_dir.find('/')
    #   poseModel = data_dir[:index]
    data = unpicklingData(POST_JSON_READING, poseModel + direction)
    print('people restored from postJsonReading.pickle')
    checkDir(poseModel)
    checkDir(poseModel + direction)
    checkDir(poseModel + direction + DATA_RESULTS)
    for n in data:
        tmp = data.get(n)
        # module for writing this data in .txt files
        checkDir(poseModel + direction + DATA_RESULTS + tmp.blockName)
        checkDir(poseModel + direction + DATA_RESULTS + tmp.blockName + '/' + tmp.videoName)
        checkDir(poseModel + direction + DATA_RESULTS + tmp.blockName + '/' + tmp.videoName + '/' + ORIGINAL)
        path2original = poseModel + direction + DATA_RESULTS + tmp.blockName + '/' + tmp.videoName + '/' + ORIGINAL
        saveData(tmp, path2original)
