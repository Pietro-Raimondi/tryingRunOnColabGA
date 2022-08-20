import os
import pickle

BODY25 = 'body25/'
COCO = 'coco/'
MPI = 'mpi/'
MPI_4 = 'mpi4/'

TEST_RAW_FRAMES = 'testRawFrames/'
POSE_FRAME = 'poseFrame/'
JSON_RAW = 'jsonRawData/'
DATASET = 'dataset/'
DIR_PICKLE = 'dirPickle/'
POST_POSE = 'postPose.pickle'
POST_JSON_READING = 'postJsonReading.pickle'
POST_PEAKS = 'postPeaks.pickle'
POST_ESTIMATION = 'postEstimation.pickle'
POST_KALMAN = 'postKalman.pickle'
FINAL_SVM_CSV_PATH = 'svmCsvPaths.pickle'
FINAL_CSV_PATH_4DBS = 'csvPaths4DbScan.pickle'

DATA_RESULTS = 'dataResults/'
STEPS = 'steps/'
ORIGINAL = 'original/'
NO_PEAKS = 'noPeaks/'
ESTIMATED = 'estimated/'
EST_POST_PEAKS = 'estPostPeaks.pickle'

CSV = 'csv/'
CSV_RAW_FEATURES = 'csv_raw/'
CSV_PROCESSED_FEATURES = 'csv_processed/'

CSV_RAW_EST = 'csv_raw/csvEstimated/'
CSV_RAW_EST_NOPEAKS = 'csv_raw/csvEstOnNoPeaks/'
CSV_RAW_KAL_EST = 'csv_raw/csvKalmanEstimated/'
CSV_RAW_KAL_EST_NOPEAKS = 'csv_raw/csvKalmanEstOnNoPeaks/'

CSV_PRCS_EST = 'csv_processed/csvEstimated/'
CSV_PRCS_EST_NOPEAKS = 'csv_processed/csvEstOnNoPeaks/'
CSV_PRCS_KAL_EST = 'csv_processed/csvKalmanEstimated/'
CSV_PRCS_KAL_EST_NOPEAKS = 'csv_processed/csvKalmanEstOnNoPeaks/'


def checkDir(pathToCheck):
    if not os.path.exists(pathToCheck):
        os.mkdir(pathToCheck)
        print("Directory ", pathToCheck, " Created ")


def picklingData(data, state, pm):
    pickle_out = open(DIR_PICKLE + pm + state, 'wb')
    pickle.dump(data, pickle_out)
    pickle_out.close()
    print('people saved as pickle file: ' + state)


def unpicklingData(state, pm):
    pickle_in = open(DIR_PICKLE + pm + state, 'rb')
    return pickle.load(pickle_in)


def analyzeModel(pm):
    if pm is 'BODY_25':
        return BODY25
    if pm is 'COCO':
        return COCO
    if pm is 'MPI':
        return MPI
    if pm is 'MPI_4_layers':
        return MPI_4
