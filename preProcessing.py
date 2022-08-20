from dataEstimation import estimatingData, estimOnNoPeaks
from dataPeaksElimination import noPeaks
from kalmanFilter import restructureData, kalmanFiltering
from dirManager import analyzeModel, unpicklingData, picklingData, POST_JSON_READING, POST_PEAKS, POST_ESTIMATION, \
    POST_KALMAN


def preprocessing(direction, pm='BODY_25'):
    poseModel = analyzeModel(pm)
    # create folder for results

    # unpickling data dictionary
    data = unpicklingData(POST_JSON_READING, poseModel + direction)
    print('people restored from postJsonReading.pickle')

    # * * * * * *  NO  PEAKS  DATA  on original data  * * * * * *
    # delete detected peaks from joints data
    noPeaks(direction, data, poseModel)
    picklingData(data, POST_PEAKS, poseModel + direction)
    print('NoPeaks data stored')

    # * * * * * *   D A T A  -  E S T I M A T I O N   * * * * * *
    # data estimation calculates missing data
    # estimation on original data
    estimatingData(direction, data, poseModel)
    picklingData(data, POST_ESTIMATION, poseModel + direction)
    print('Estimated data on original data stored')

    # estimation on data without peaks
    for n in data:
        tmp = data.get(n)
        p, v = estimOnNoPeaks(tmp.withoutPeaks.pose2d, tmp.withoutPeaks.visibilities)
        tmp.estOnNoPeaks.append(p)
        tmp.estOnNoPeaks.append(v)
    picklingData(data, POST_ESTIMATION, poseModel + direction)
    print('Estimated data on data without peaks stored')

    # * * * * * *   K A L M A N  -  F I L T E R   * * * * * *
    # Kalman Filter can be applied only on complete data(all joints, all frames)

    # KALMAN FILTER ON JUST ESTIMATED DATA
    restructureData(data, 'estimated')
    kalmanFiltering(data, 'estimated')
    print('Kalman applied on estimated data')

    # KALMAN FILTER ON PREPROCESSED DATA WITH ALL PREVIOUS ALGORITHM
    restructureData(data, 'estOnNoPeaks')
    kalmanFiltering(data, 'estOnNoPeaks')
    print('Kalman applied on estOnNoPeaks data')

    picklingData(data, POST_KALMAN, poseModel + direction)
