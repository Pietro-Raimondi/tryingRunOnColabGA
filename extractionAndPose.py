import argparse
import sys
import time
import cv2
import re
import os
from dataVideoClass import DataVideoTest
from framesExtraction import testExtraction
from dirManager import checkDir, TEST_RAW_FRAMES, analyzeModel, POSE_FRAME, JSON_RAW, DIR_PICKLE, POST_POSE, \
    picklingData

# DATASET = 'dataset/'

# sorting module functions
# ********************  S T A R T  *********************
def tryInt(s):
    try:
        return int(s)
    except ValueError:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryInt(c) for c in re.split('([0-9]+)', s)]
# *********************  E N D  ********************


def importOP():
    path = os.path.dirname(os.path.realpath(__file__))
    try:
        sys.path.append(path + '/python/openpose/Release')
        os.environ['PATH'] = os.environ['PATH'] + ';' + path + '/x64/Release;' + path + '/bin;'
        # sys.path.append(path + '/python/openpose/Release')
        # os.environ['PATH'] = os.environ['PATH'] + ';' + path + '/x64/Release;' + path + '/bin;'
        import pyopenpose as op
        return op, path
    except ImportError as err:
        print('Error: OpenPose library could not be found. '
              'Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise err


def extrAndPose(direction, poseModel='BODY_25'):
    #  EXTRACTION  AND  POSE ESTIMATION
    # import openPose lib
    totalStart = time.time()
    op, dir_path = importOP()
    try:
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
        # dict key:videoName, value: videoData
        data = {}
        # pm = None
        # create general raw frames folder
        checkDir(TEST_RAW_FRAMES)

        # WALKING DATASET
        for root, dirs, files in os.walk('datatest'+direction):
            dirs.sort(key=alphanum_key)
            if len(files) is not 0:
                # pm = poseModel
                files.sort(key=alphanum_key)
                for file in files:
                    pm = poseModel
                    # this two lines return the block's name
                    blockName = os.path.splitext(root)[0]
                    blockName = blockName.split("/")[-1]
                    blockName = blockName.split("\\")[0]
                    # create dataStructure for video
                    dataTest = DataVideoTest()
                    dataTest.blockName = blockName
                    dataTest.videoName = os.path.splitext(os.path.basename(file))[0]
                    checkDir(TEST_RAW_FRAMES + direction)
                    checkDir(TEST_RAW_FRAMES + direction + blockName)
                    dataTest.pathToFile = root + '/' + dataTest.videoName + '.avi'
                    dataTest.pathRawFrames = TEST_RAW_FRAMES + direction + dataTest.blockName + '/' + dataTest.videoName + '/'
                    # create relative raw frame folder
                    checkDir(dataTest.pathRawFrames)
                    print('********************************************************* START ' + dataTest.videoName)
                    # * * * * * *  E X T R A C T I O N  * * * * * *
                    # if dataTest.blockName == '/p26s1':
                    testExtraction(dataTest)

                    # frame sorting module
                    # sorting frames
                    obj = os.listdir(dataTest.pathRawFrames)
                    obj.sort(key=alphanum_key)

                    # * * * * * *  P O S E   E S T I M A T I O N  * * * * * *
                    parser = argparse.ArgumentParser()
                    parser.add_argument("--image_dir", default=dir_path + '/' + dataTest.pathRawFrames,
                                        help="Process a directory of images. "
                                             "Read all standard formats (jpg, png, bmp, etc.).")
                    parser.add_argument("--no_display", default=True, help="Enable to disable the visual display.")
                    args = parser.parse_known_args()
                    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
                    params = dict()
                    params['number_people_max'] = 1
                    params['profile_speed'] = 10
                    params['model_folder'] = 'models/'
                    params['model_pose'] = pm
                    # params['model_pose'] = 'COCO'
                    # params['model_pose'] = 'MPI'
                    # params['model_pose'] = 'MPI_4_layers'
                    # resolution from faster to slower, from less to more accurate
                    # params['net_resolution'] = '432x368'
                    params['net_resolution'] = '880x480'
                    # params["net_resolution"] = '928x528'

                    pm = analyzeModel(pm)
                    checkDir(pm)

                    checkDir(pm + direction)

                    # create pose frame folder for each video
                    checkDir(pm + direction + POSE_FRAME)
                    checkDir(pm + direction + POSE_FRAME + dataTest.blockName)
                    checkDir(pm + direction + POSE_FRAME + dataTest.blockName + '/' + dataTest.videoName + '/')
                    dataTest.pathToPose = pm + direction + POSE_FRAME + dataTest.blockName + '/' + dataTest.videoName + '/'
                    # create raw json data folder for each video
                    checkDir(pm + direction + JSON_RAW)
                    checkDir(pm + direction + JSON_RAW + dataTest.blockName)
                    checkDir(pm + direction + JSON_RAW + dataTest.blockName + '/' + dataTest.videoName + '/')
                    dataTest.pathToJson = pm + direction + JSON_RAW + dataTest.blockName + '/' + dataTest.videoName + '/'

                    params['write_json'] = dataTest.pathToJson
                    params['write_images'] = dataTest.pathToPose
                    # Starting OpenPose
                    opWrapper = op.WrapperPython()
                    opWrapper.configure(params)
                    opWrapper.start()
                    # Read frames on directory
                    imagePaths = op.get_images_on_directory(args[0].image_dir)
                    start = time.time()  # comment4gpu from here
                    # Process and display images
                    count = 0
                    for imagePath in imagePaths:
                        datum = op.Datum()
                        imageToProcess = cv2.imread(imagePath)
                        datum.cvInputData = imageToProcess
                        opWrapper.emplaceAndPop([datum])
                        print('frame ' + str(count))
                        count += 1
                    end = time.time()
                    print('OpenPose on ' + dataTest.blockName + '/' + dataTest.videoName +
                          'successfully finished. Total time: ' + str(end - start) + 'seconds')
                    print('********************************************************* END ' + dataTest.videoName)
                    # add data single video to dictionary
                    # DataVideoTest2 is needed for pickling the Class

                    # dataTest2 = DataVideoTest2(dataTest)
                    # data[dataTest2.videoName] = dataTest2

                    data[dataTest.blockName + '-' + dataTest.videoName] = dataTest
        checkDir(DIR_PICKLE)
        pm = analyzeModel(poseModel)
        checkDir(DIR_PICKLE + pm)
        checkDir(DIR_PICKLE + pm + direction)
        picklingData(data, POST_POSE, pm + direction)
        totalEnd = time.time()
        print('successfully finished. Total time: ' + str(totalEnd - totalStart) + 'seconds')
    except Exception as e:
        print(e)
        sys.exit(-1)
