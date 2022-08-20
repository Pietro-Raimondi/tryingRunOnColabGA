from dirManager import analyzeModel, unpicklingData, checkDir, BODY25, POST_KALMAN, CSV, CSV_RAW_FEATURES, CSV_RAW_EST, \
    CSV_RAW_EST_NOPEAKS, CSV_RAW_KAL_EST, CSV_RAW_KAL_EST_NOPEAKS
from featureExtraction import toDict4Extraction, extract_and_store_features


def creatingFeatures(direction, poseModel='BODY_25'):
    pm = analyzeModel(poseModel)
    data = unpicklingData(POST_KALMAN, BODY25 + direction)

    # 'csv_raw/'
    checkDir(pm + direction + CSV)
    checkDir(pm + direction + CSV+CSV_RAW_FEATURES)
    checkDir(pm + direction + CSV+CSV_RAW_EST)
    checkDir(pm + direction + CSV+CSV_RAW_EST_NOPEAKS)
    checkDir(pm + direction + CSV+CSV_RAW_KAL_EST)
    checkDir(pm + direction + CSV+CSV_RAW_KAL_EST_NOPEAKS)

    for n in data:
        tmp = data.get(n)

        # creating relatives folders for each type of algorithm applied to data
        checkDir(pm + direction + CSV+CSV_RAW_EST + tmp.blockName + '/')
        pathToCSV1 = pm + direction + CSV+CSV_RAW_EST + tmp.blockName + '/'
        checkDir(pathToCSV1)

        checkDir(pm + direction + CSV+CSV_RAW_EST_NOPEAKS + tmp.blockName + '/')
        pathToCSV2 = pm + direction + CSV+CSV_RAW_EST_NOPEAKS + tmp.blockName + '/'
        checkDir(pathToCSV2)

        checkDir(pm + direction + CSV+CSV_RAW_KAL_EST + tmp.blockName + '/')
        pathToCSV3 = pm + direction + CSV+CSV_RAW_KAL_EST + tmp.blockName + '/'
        checkDir(pathToCSV3)

        checkDir(pm + direction + CSV+CSV_RAW_KAL_EST_NOPEAKS + tmp.blockName + '/')
        pathToCSV4 = pm + direction + CSV+CSV_RAW_KAL_EST_NOPEAKS + tmp.blockName + '/'
        checkDir(pathToCSV4)

        times = tmp.person.times

        # extract specific data from chosen Dict and save raw features as CSV file in passed path
        print('WORKING ON ESTIMATED VALUES')
        estimatedDict = toDict4Extraction(tmp.estimated.pose2d, tmp.estimated.visibilities)
        extract_and_store_features(estimatedDict, times,
                                   pathToCSV1, tmp.videoName)

        print('WORKING ON ESTIMATED + NO PEAKS VALUES')
        estOnNoPeaksDict = toDict4Extraction(tmp.estOnNoPeaks[0], tmp.estOnNoPeaks[1])
        extract_and_store_features(estOnNoPeaksDict, times,
                                   pathToCSV2, tmp.videoName)

        print('WORKING ON ESTIMATED + KALMAN VALUES')
        extract_and_store_features(tmp.kalmanEstimated[1], times,
                                   pathToCSV3, tmp.videoName)

        print('WORKING ON ESTIMATED + NO PEAKS + KALMAN VALUES')
        extract_and_store_features(tmp.kalmanEstOnNoPeaks[1], times,
                                   pathToCSV4, tmp.videoName)
