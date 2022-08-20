import os
import re
from dirManager import CSV, analyzeModel, CSV_PRCS_KAL_EST_NOPEAKS, CSV_PRCS_KAL_EST, CSV_PRCS_EST_NOPEAKS, \
    CSV_PRCS_EST, picklingData, FINAL_CSV_PATH_4DBS
import pandas as pd


def joiningCsvToDf(direction, csvType, pm='BODY_25'):

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
                tmpDataFrame = pd.read_csv(completePathName)
                finalDataset = pd.DataFrame(columns=tmpDataFrame.columns)
                finalDataset["Target"] = 0
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
                tmpDataFrame = pd.read_csv(completePathName)
                tmpDataFrame['Target'] = numberPerson
                finalDataset = pd.concat([finalDataset, tmpDataFrame], ignore_index=True)
                # finalDataset = finalDataset.append(tmpDataFrame)
    return finalDataset


def fromDfToCsv(df, n, direction):
    df.to_csv(os.getcwd() + '/body25/' + direction + n + '4DbScan' + '.csv')
    path = os.getcwd() + '/body25/' + direction + n + '4DbScan' + '.csv'
    return path


def printing(df):
    print(df.shape)
    # print(df.head())
    # print(df.describe())
    # iterating columns, prints index of rows
    # for row in df.index:
    #     print(row, end=" ")


def joiningAllDatasets4DbScan(direction, pm='BODY_25'):
    path2CsvFiles4DbScan = []
    pm = analyzeModel('BODY_25')
    # print(os.getcwd())
    # both = '/both/'
    print('one')
    estimatedDf = joiningCsvToDf(direction, CSV_PRCS_EST, pm)
    printing(estimatedDf)
    path2CsvFiles4DbScan.append(fromDfToCsv(estimatedDf, 'est', direction))

    print('two')
    estNoPeaksDf = joiningCsvToDf(direction, CSV_PRCS_EST_NOPEAKS, pm)
    printing(estNoPeaksDf)
    path2CsvFiles4DbScan.append(fromDfToCsv(estNoPeaksDf, 'estNoPeaks', direction))

    print('three')
    kalmanEstDf = joiningCsvToDf(direction, CSV_PRCS_KAL_EST, pm)
    printing(kalmanEstDf)
    path2CsvFiles4DbScan.append(fromDfToCsv(kalmanEstDf, 'kalmanEst', direction))

    print('four')
    kalmanEstNoPeaksDf = joiningCsvToDf(direction, CSV_PRCS_KAL_EST_NOPEAKS, pm)
    printing(kalmanEstNoPeaksDf)
    path2CsvFiles4DbScan.append(fromDfToCsv(kalmanEstNoPeaksDf, 'kalmanEstNoPeaks', direction))

    picklingData(path2CsvFiles4DbScan, FINAL_CSV_PATH_4DBS, pm + direction)

'''
if __name__ == "__main__":
    path2CsvFiles4DbScan = []
    pm = analyzeModel('BODY_25')
    # print(os.getcwd())
    both = '/both/'
    print('one')
    estimatedDf = joiningCsvToDf(both, CSV_PRCS_EST, pm)
    printing(estimatedDf)
    path2CsvFiles4DbScan.append(fromDfToCsv(estimatedDf, 'est', both))

    print('two')
    estNoPeaksDf = joiningCsvToDf(both, CSV_PRCS_EST_NOPEAKS, pm)
    printing(estNoPeaksDf)
    path2CsvFiles4DbScan.append(fromDfToCsv(estNoPeaksDf, 'estNoPeaks', both))

    print('three')
    kalmanEstDf = joiningCsvToDf(both, CSV_PRCS_KAL_EST, pm)
    printing(kalmanEstDf)
    path2CsvFiles4DbScan.append(fromDfToCsv(kalmanEstDf, 'kalmanEst', both))

    print('four')
    kalmanEstNoPeaksDf = joiningCsvToDf(both, CSV_PRCS_KAL_EST_NOPEAKS, pm)
    printing(kalmanEstNoPeaksDf)
    path2CsvFiles4DbScan.append(fromDfToCsv(kalmanEstNoPeaksDf, 'kalmanEstNoPeaks', both))

    picklingData(path2CsvFiles4DbScan, FINAL_CSV_PATH_4DBS, pm + both)
'''