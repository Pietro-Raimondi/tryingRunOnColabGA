import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # loading dataset from csv file
    #   dataframe = pd.read_csv('body25/csv/csv_raw/csvEstimated/p1s1/c1_0195.csv')

    dataframe = pd.read_csv('body25/csv/csv_raw/csvEstOnNoPeaks/p14s1/estOnNoPeaksc1_0005.csv')
    a = dataframe.to_dict('list')
    for n in a:
        a.get(n)
    print(dataframe.shape)
    print(dataframe.head(5))
    print(dataframe.describe())
    k = []
    for label in dataframe.columns:
        k.append(label)
    print('wait')
