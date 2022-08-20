import os
import re
import pandas as pd
from extractionAndPose import analyzeModel
from dirManager import CSV, CSV_RAW_EST, CSV_RAW_EST_NOPEAKS, CSV_RAW_KAL_EST, CSV_RAW_KAL_EST_NOPEAKS, picklingData, \
    FINAL_SVM_CSV_PATH


def joiningCsvToDf(csvType, direction, pm):
    # pm = analyzeModel(poseModel)
    dfCreated = False
    finalDataset = pd.DataFrame

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

    for root, dirs, files in os.walk(pm + direction + CSV + csvType):
        dirs.sort(key=alphanum_key)
        if len(files) is not 0:
            files.sort(key=alphanum_key)

            for file in files:
                # print(file)
                completePathName = root + '/' + file
                # print(completePathName)
                tmpDataFrame = pd.read_csv(completePathName, index_col=0)
                # tmpDataFrame2 = pd.read_csv(completePathName)
                # print(tmpDataFrame.head())
                # print(tmpDataFrame2.head())
                finalDataset = pd.DataFrame(columns=tmpDataFrame.columns)
                # print(finalDataset.head())
                finalDataset["Target"] = 0
                # print(finalDataset.head())
                dfCreated = True
            if dfCreated:
                break

    for root, dirs, files in os.walk(pm + direction + CSV + csvType):
        dirs.sort(key=alphanum_key)
        if len(files) is not 0:
            files.sort(key=alphanum_key)
            blockName = os.path.splitext(root)[0].split('/')[-1]
            numberPerson = blockName.split('s')[0].split('p')[-1]
            # print(numberPerson)

            for file in files:
                completePathName = root + '/' + file
                tmpDataFrame = pd.read_csv(completePathName, index_col=0)
                # print(tmpDataFrame.head())
                tmpDataFrame['Target'] = int(numberPerson)
                # print(tmpDataFrame.head())
                finalDataset = pd.concat([finalDataset, tmpDataFrame], ignore_index=True)
                # print(tmpDataFrame.head())
                # finalDataset = finalDataset.append(tmpDataFrame)
    return finalDataset


def fromDfToCsv(df, direction, name):
    df.to_csv(os.getcwd() + '/body25/' + direction + name + '.csv')

    # df.to_csv(os.getcwd() + '/body25/' + direction + name + '.csv', index=False)
    # print(df.head())
    # tmpDataFrame = pd.read_csv((os.getcwd() + '/body25/' + direction + name + '.csv'))
    # print(tmpDataFrame.head())
    path = os.getcwd() + '/body25/' + direction + name + '.csv'
    return path


def printing(df):
    print(df.shape)
    print(df.head(5))
    # print(df.describe())
    # iterating columns, prints index of rows
    # for row in df.index:
    #     print(row, end=" ")


def joiningAllDatasets(direction, poseModel='BODY_25'):
    path2CsvFiles = []
    pm = analyzeModel(poseModel)
    # print(os.getcwd())
    print('estimated' + direction)
    estimatedDf = joiningCsvToDf(CSV_RAW_EST, direction, pm)
    printing(estimatedDf)
    # print(estimatedDf.head())
    path2CsvFiles.append(fromDfToCsv(estimatedDf, direction, 'estimatedDf'))

    print('estNoPeaks' + direction)
    estNoPeaksDf = joiningCsvToDf(CSV_RAW_EST_NOPEAKS, direction, pm)
    printing(estNoPeaksDf)
    path2CsvFiles.append(fromDfToCsv(estNoPeaksDf, direction, 'estNoPeaksDf'))

    print('kalmanEst' + direction)
    kalmanEstDf = joiningCsvToDf(CSV_RAW_KAL_EST, direction, pm)
    printing(kalmanEstDf)
    path2CsvFiles.append(fromDfToCsv(kalmanEstDf, direction, 'kalmanEstDf'))

    print('kalmanEstNoPeaks' + direction)
    kalmanEstNoPeaksDf = joiningCsvToDf(CSV_RAW_KAL_EST_NOPEAKS, direction, pm)
    printing(kalmanEstNoPeaksDf)
    path2CsvFiles.append(fromDfToCsv(kalmanEstNoPeaksDf, direction, 'kalmanEstNoPeaksDf'))
    # return path2CsvFiles

    picklingData(path2CsvFiles, FINAL_SVM_CSV_PATH, pm + direction)
    print('paths for %s data stored in csv files for SVMachine' % direction[1:-1])


"""
if __name__ == "__main__":
    # print(os.getcwd())
    print('one')
    estimatedDf = joiningCsvToDf(CSV_RAW_EST, direction, 'BODY_25')
    printing(estimatedDf)
    fromDfToCsv(estimatedDf, 'estimatedDf')

    print('\ntwo')
    estNoPeaksDf = joiningCsvToDf(CSV_RAW_EST_NOPEAKS, direction, 'BODY_25')
    printing(estNoPeaksDf)
    fromDfToCsv(estNoPeaksDf, 'estNoPeaksDf')

    print('\nthree')
    kalmanEstDf = joiningCsvToDf(CSV_RAW_KAL_EST, direction, 'BODY_25')
    printing(kalmanEstDf)
    fromDfToCsv(kalmanEstDf, 'kalmanEstDf')

    print('\nfour')
    kalmanEstNoPeaksDf = joiningCsvToDf(CSV_RAW_KAL_EST_NOPEAKS, direction, 'BODY_25')
    printing(kalmanEstNoPeaksDf)
    fromDfToCsv(kalmanEstNoPeaksDf, 'kalmanEstNoPeaksDf')

    # this lines can return a specific part of dataframe: choosing specific values of specific columns
    # print(estimatedDf['Target'])
    # justFirstPersonData = estimatedDf[estimatedDf["Target"] == 4]
    # print(justFirstPersonData.head(5))
    
"""
