import time
import pandas as pd
from sklearn import datasets

from decisionTree import decisionTree
from joiningDf4Clustering import joiningAllDatasets4DbScan
from joiningFeaturesDf4Classification import joiningAllDatasets
from dirManager import unpicklingData, POST_JSON_READING, FINAL_SVM_CSV_PATH, analyzeModel, FINAL_CSV_PATH_4DBS
from extractionAndPose import extrAndPose
from creatingTemporalSpaceFeatures import creatingFeatures
from creatingOneLineFeatures import processingFeatures
from preProcessing import preprocessing
from randomForest import randomForest
from readDataFromJson import readDataFromJson
from regressioneLineare import linearRegression
from storingModule import saveOriginalData
from testSVM import svmMachine


def runThemAll(direction, pm):
    # # load iris dataset
    # iris = datasets.load_iris()
    # # Since this is a bunch, create a dataframe
    # iris_df = pd.DataFrame(iris.data)
    # iris_df['class'] = iris.target

    # Extraction and pose estimation; dataVideo saved in .pickle file
    # extrAndPose(direction, pm)

    # Reading data from json and storing into personStruct for each videoDataClass; dataVideo saved in .pickle file
    # readDataFromJson(direction, pm)

    # Reading data from personStruct and writing info in .txt files inside relative 'dataResults' folder
    # saveOriginalData(direction, pm)

    # Preprocessing on data (noPeaks/estimation/KalmanFilter), furthermore this data are saved in .txt files and pickled
    # preprocessing(direction, pm)

    # Create features and store them in .csv files inside relative 'csv_raw' folder for each type of data
    # creatingFeatures(direction, pm)

    # Maybe useless
    # processingFeatures(direction, pm)

    # Create single dataFrame, saved in csv file, containing all people, for each Person there are all frames
    # for each frame there are all joints with relative measures, add target label for each of them
    # finally the list with this paths will be saved in pickle file
    # joiningAllDatasets(direction, pm)

    # same thing but for the DbScan Algorithm
    # joiningAllDatasets4DbScan(direction, pm)

    # path = 'C:\\Users\\raimo\\Desktop\\projSides/body25//dx2sx/estimatedDf.csv'
    # svmMachine(path)
    # this method applies the svm to 4 dataFrame read by the passed path to csv files
    # '''


    pm = analyzeModel(pm)
    path2CsvSVMFiles = unpicklingData(FINAL_SVM_CSV_PATH, pm + direction)
    # path2csvDbScanFiles = unpicklingData(FINAL_CSV_PATH_4DBS, pm + direction)

    for path in path2CsvSVMFiles:
    # for path in path2csvDbScanFiles:
        # decisionTree(path)
        # linearRegression(path)
        svmMachine(path)
        # randomForest(path)


if __name__ == '__main__':

    # SELECT MODEL POSE (standard='BODY_25', 'COCO', 'MPI', 'MPI_4_layers')
    pm = 'BODY_25'
    # direction
    sx2dx = '/sx2dx/'
    dx2sx = '/dx2sx/'
    both = '/both/'
    sides = [sx2dx, dx2sx, both]
    start = time.time()
    for side in sides:
        runThemAll(side, pm)
    # runThemAll(sx2dx, pm)
    # runThemAll(dx2sx, pm)
    # runThemAll(both, pm)
    end = time.time()
    print('total time needed: ' + str(end - start) + 'seconds')
