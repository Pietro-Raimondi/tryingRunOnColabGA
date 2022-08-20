import sklearn
from sklearn import datasets, svm, metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from mlxtend.plotting import plot_decision_regions

from dirManager import analyzeModel, unpicklingData, FINAL_SVM_CSV_PATH


def linearRegression(path2CsvFiles):
    # df = pd.read_csv(path2CsvFiles[0])
    df = pd.read_csv(path2CsvFiles, index_col=0)
    print(f'number of rows/examples and columns in the dataset: {df.shape}')
    print(path2CsvFiles)
    X = df.drop('Target', axis=1)
    y = df['Target']

    X = np.nan_to_num(X)
    sc = StandardScaler()
    X_std = sc.fit_transform(X)

    pca = PCA()
    pca.fit(X_std)

    X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.20, random_state=42)

    lr = LogisticRegression(C=10, max_iter=20000, n_jobs=-1, solver='sag', verbose=3)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    # print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    # print(confusion_matrix(y_test, y_pred, labels=[1, 0]).reshape(-1))
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred, labels=lr.classes_), display_labels=lr.classes_).plot()
    plt.show()


if __name__ == '__main__':
    poseModel = analyzeModel('BODY_25')
    sx2dx = '/sx2dx/'
    dx2sx = '/dx2sx/'
    both = '/both/'
    path2CsvFiles = unpicklingData(FINAL_SVM_CSV_PATH, poseModel + sx2dx)
    for path in path2CsvFiles:
        linearRegression(path)
