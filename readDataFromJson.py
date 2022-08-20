import json
import os
from extractionAndPose import alphanum_key, analyzeModel
from dirManager import POST_POSE, POST_JSON_READING, picklingData, unpicklingData
from dataVideoClass import Person


def readDataFromJson(direction, poseModel='BODY_25'):
    pm = analyzeModel(poseModel)
    # unpickling data dictionary
    data = unpicklingData(POST_POSE, pm + direction)
    print('people restored from afterPose.pickle')
    for n in data:
        pose2d, visibilities, time = [], [], []
        tmp = data.get(n)
        json_files = os.listdir(tmp.pathToJson)
        # sorting list of json files
        json_files.sort(key=alphanum_key)
        # cycle to analyze each json file
        for j in range(0, len(json_files)):
            # loading json file
            with open(tmp.pathToJson + json_files[j], 'r') as f:
                loaded_json = json.load(f)
            for person in loaded_json['people']:
                coordinates = person['pose_keypoints_2d']
            # array in which to store x and y coordinates of each body part of selected person in the frame
            frame_coords = []
            # array in which to store visibilities of each body part of selected person in the frame
            frame_visib = []
            # cycle to analize each information of interest in data array
            i = 0
            while i < 45:
                # storing x and y coordinates and visibilities of each body part, avoiding the coordinates of middle hip
                if i != 24:
                    frame_coords.append([int(coordinates[i]), int(coordinates[i + 1])])
                    if coordinates[i] == 0:
                        frame_visib.append(False)
                    else:
                        frame_visib.append(True)
                i = i + 3
            pose2d.append(frame_coords)
            visibilities.append(frame_visib)
            time.append(j / tmp.fps)
        tmp.person = Person(pose2d, visibilities, time)
        a, b, c, d, e, f = [], [], [], [], [], []
        tmp.withoutPeaks = Person(a, b, c)
        tmp.estimated = Person(d, e, f)
        # end of module for storing data into structure Person
    picklingData(data, POST_JSON_READING, pm + direction)
