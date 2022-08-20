import os
import numpy as np
import math
from typing import Iterable
import pandas as pd
from dirManager import checkDir, CSV, analyzeModel, CSV_PROCESSED_FEATURES, CSV_PRCS_EST, CSV_PRCS_EST_NOPEAKS, \
    CSV_PRCS_KAL_EST, CSV_PRCS_KAL_EST_NOPEAKS, CSV_RAW_EST, CSV_RAW_EST_NOPEAKS, CSV_RAW_KAL_EST, \
    CSV_RAW_KAL_EST_NOPEAKS


# START: utility functions

def feature_mean(array):
    return np.mean(array.copy(), axis=0)


def feature_median(array):
    return np.median(array.copy(), axis=0)


def feature_standard_deviation(array):
    return np.std(array.copy(), axis=0)


def feature_one_percentile(array):
    #   a = np.percentile(array.copy(), 1)
    a = percentile(array.copy(), 1)
    return a


def feature_ninetyNine_percentile(array):
    a = np.percentile(array.copy(), 99)
    # a = percentile(array.copy(), 99)
    return a


# def percentile(n: Iterable, percent: int):
def percentile(n, percent: int):
    """
    Find the percentile of a list of values.
    Stolen from http://code.activestate.com/recipes/511478-finding-the-percentile-of-the-values/
    """
    if type(n) is np.ndarray:
        n = n.tolist()
    if not n:
        return 0

    if not (0 < percent < 100 and type(percent) == int):
        raise ValueError('percent parameter must be integer from 0 to 100')

    n.sort()

    k = (len(n) - 1) * (percent/100)
    prev_index = math.floor(k)
    next_index = math.ceil(k)

    if prev_index == next_index:
        return n[int(k)]

    d0 = n[prev_index] * (next_index - k)
    d1 = n[next_index] * (k - prev_index)

    return d0 + d1
# END: utility functions


def calculatingFeaturesFromDF(dataFrame):
    personFeatures = {}
    for column in dataFrame:
        tmp = dataFrame[column].tolist()
        personFeatures[column + '-Mean'] = np.round(feature_mean(tmp), 2)
        personFeatures[column + '-Median'] = np.round(feature_median(tmp), 2)
        personFeatures[column + '-StdDev'] = np.round(feature_standard_deviation(tmp), 2)
        personFeatures[column + '-1Pctl'] = np.round(feature_one_percentile(tmp), 2)
        personFeatures[column + '-99Pctl'] = np.round(feature_ninetyNine_percentile(tmp), 2)
    return personFeatures


# creating final dataframe features
def processingAllFeatures(rawFeaturesFolder, destinationFolder):
    for root, dirs, files in os.walk(rawFeaturesFolder):
        dirs.sort()
        if len(files) is not 0:
            files.sort()
            # this two lines return the block's name
            blockName = os.path.splitext(root)[0]
            blockName = blockName[len(rawFeaturesFolder):]
            # create block video folder for raw frames
            checkDir(destinationFolder + blockName)
            for file in files:
                # videoName = os.path.splitext(os.path.basename(file))[0]
                pathToFile = root + '/' + file
                # loading dataframe from csv file
                dataframe = pd.read_csv(pathToFile)
                # calculating features
                newDict = calculatingFeaturesFromDF(dataframe)
                # turning dict in dataFrame
                df = pd.DataFrame(newDict, index=[0])
                # saving dataFrame in csv file to relative destination folder
                df.to_csv(destinationFolder + blockName + '/' + file, index=False)


def processingFeatures(direction, poseModel='BODY_25'):
    pm = analyzeModel(poseModel)

    checkDir(pm + direction + CSV+CSV_PROCESSED_FEATURES)
    checkDir(pm + direction + CSV+CSV_PRCS_EST)
    checkDir(pm + direction + CSV+CSV_PRCS_EST_NOPEAKS)
    checkDir(pm + direction + CSV+CSV_PRCS_KAL_EST)
    checkDir(pm + direction + CSV+CSV_PRCS_KAL_EST_NOPEAKS)

    # all raw features have been found. Now we'll find mean, standard deviation, median, 1 percentile and 99 percentile
    # this calculation will be done for every joint, and every coordinate joint
    # once all the features are computed, the function will save them as csv file into relatives folders
    processingAllFeatures(pm + direction + CSV+CSV_RAW_EST, pm + direction + CSV+CSV_PRCS_EST)
    processingAllFeatures(pm + direction + CSV+CSV_RAW_EST_NOPEAKS, pm + direction + CSV+CSV_PRCS_EST_NOPEAKS)
    processingAllFeatures(pm + direction + CSV+CSV_RAW_KAL_EST, pm + direction + CSV+CSV_PRCS_KAL_EST)
    processingAllFeatures(pm + direction + CSV+CSV_RAW_KAL_EST_NOPEAKS, pm + direction + CSV+CSV_PRCS_KAL_EST_NOPEAKS)
