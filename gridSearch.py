import os

import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from dirManager import analyzeModel, unpicklingData, FINAL_SVM_CSV_PATH, checkDir


def logisticRegressionGrid(X_train, X_test, y_train, y_test, side, i):
    path = 'C:/Users/raimo/Desktop/lastPythonProject/'
    logClassifier = LogisticRegression()
    # define evaluation
    # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # define search space
    '''
    space = dict()
    space['solver'] = ['liblinear', 'saga', 'newton-cg']
    # space['solver'] = ['newton-cg', 'lbfgs', 'liblinear', 'svd', 'cholesky', 'lsqr', 'sag']
    # space['penalty'] = ['l1', 'l2', 'elasticnet']
    space['C'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    space['penalty'] = ['l2']
    space['max_iter'] = [10000, 20000, 30000]'''
    '''    
    space['solver'] = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    # space['solver'] = ['newton-cg', 'lbfgs', 'liblinear', 'svd', 'cholesky', 'lsqr', 'sag']
    space['penalty'] = ['l1', 'l2', 'elasticnet']
    space['C'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 50, 100]
    space['max_iter'] = [20000]'''
    # define search
    # search = GridSearchCV(model, space, scoring='accuracy', n_jobs=-1, cv=5)
    param_grid = {'solver': ['liblinear', 'saga', 'newton-cg'],
                  'C': [1e-1, 1, 5],
                  'max_iter': [10000, 20000, 30000],
                  'penalty': ['l2'],
                  }
    grid = GridSearchCV(logClassifier, param_grid, scoring='accuracy', n_jobs=4, verbose=3)

    # execute search
    grid.fit(X_train, y_train)
    results = [grid]

    # summarize result
    print('Best Score: %s' % grid.best_score_)
    print('Best Hyperparameters: %s' % grid.best_params_)

    # Print the test score of the best model
    clfRFC = grid.best_estimator_
    print('Test accuracy: %.3f' % clfRFC.score(X_test, y_test))

    # who knows?
    grid_predictions = grid.predict(X_test)
    print(classification_report(y_test, grid_predictions))

    results.append(grid_predictions)
    elem = [X_train, X_test, y_train, y_test, side]
    results.append(elem)
    pickle_out = open(path + 'results' + side + 'RegressionGrid'+str(i)+'.pickle', 'wb')
    pickle.dump(results, pickle_out)
    pickle_out.close()
    print('people saved as pickle file')


def decisionTreeGrid(X_train, X_test, y_train, y_test, side, i):
    path = 'C:/Users/raimo/Desktop/lastPythonProject/'

    param_grid = {'max_features': ['auto', 'sqrt', 'log2'],
                  'splitter': ['best', 'random'],
                  'ccp_alpha': [0.1, 0.001, 0.0001, 0.00001],
                  'max_depth': [1, 15, 50, 100, 1000, None],
                  'criterion': ['gini', 'entropy']
                  }
    tree_clas = DecisionTreeClassifier(random_state=42)
    grid = GridSearchCV(estimator=tree_clas, param_grid=param_grid, n_jobs=3, verbose=3)
    grid.fit(X_train, y_train)

    results = [grid]

    # summarize result
    print('Best Score: %s' % grid.best_score_)
    print('Best Hyperparameters: %s' % grid.best_params_)

    # Print the test score of the best model
    clfRFC = grid.best_estimator_
    print('Test accuracy: %.3f' % clfRFC.score(X_test, y_test))

    # who knows?
    grid_predictions = grid.predict(X_test)
    print(classification_report(y_test, grid_predictions))

    results.append(grid_predictions)
    elem = [X_train, X_test, y_train, y_test, side]
    results.append(elem)
    pickle_out = open(path + 'results' + side + 'DecisionTree'+str(i)+'.pickle', 'wb')
    pickle.dump(results, pickle_out)
    pickle_out.close()
    print('people saved as pickle file')


def randomForestGrid(X_train, X_test, y_train, y_test, side, i):
    path = 'C:/Users/raimo/Desktop/lastPythonProject/'

    n_estimators = [200, 500, 800]
    # Number of features to consider at every split
    max_features = [None, 'auto', 'sqrt']
    max_depth = [70, 100, None]

    # Minimum number of samples required to split a node
    # min_samples_split = [2, 5, 10]

    # Minimum number of samples required at each leaf node
    # min_samples_leaf = [1, 2, 4]

    # Method of selecting samples for training each tree
    # bootstrap = [True, False]
    bootstrap = [False]

    # Create the random grid
    # param_grid = {'criterion': ['gini', 'entropy'],
    #                'n_estimators': n_estimators,
    #                'max_features': max_features,
    #                'max_depth': max_depth,
    #                'min_samples_split': min_samples_split,
    #                'min_samples_leaf': min_samples_leaf,
    #                'bootstrap': bootstrap}
    param_grid = {'criterion': ['gini', 'entropy'],
                  'n_estimators': n_estimators,
                  'max_features': max_features,
                  'max_depth': max_depth,
                  'bootstrap': bootstrap}

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rfc = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(estimator=rfc, param_grid=param_grid, n_jobs=4, verbose=3)
    grid.fit(X_train, y_train)

    results = [grid]

    # summarize result
    print('Best Score: %s' % grid.best_score_)
    print('Best Hyperparameters: %s' % grid.best_params_)

    # Print the test score of the best model
    clfRFC = grid.best_estimator_
    print('Test accuracy: %.3f' % clfRFC.score(X_test, y_test))

    # who knows?
    grid_predictions = grid.predict(X_test)
    print(classification_report(y_test, grid_predictions))

    results.append(grid_predictions)
    elem = [X_train, X_test, y_train, y_test, side]
    results.append(elem)
    pickle_out = open(path + 'results' + side + 'RandomForest'+str(i)+'.pickle', 'wb')
    pickle.dump(results, pickle_out)
    pickle_out.close()
    print('people saved as pickle file')


def svmGrid(X_train, X_test, y_train, y_test, side, i):
    path = 'C:/Users/raimo/Desktop/lastPythonProject/'

    # defining parameter range
    # param_grid = {'C': [0.1, 1, 10, 100, 1000],
    #               'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    #               'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}

    param_grid = {'C': [100000, 10000, 1000, 100, 1],
                  'gamma': ['auto', 'scale'],
                  'kernel': ['rbf']}

    # param_grid = {'C': [10],
    #               'gamma': ['scale'],
    #               'kernel': ['rbf']}

    # grid = GridSearchCV(SVC(), param_grid, refit=True, n_jobs=-1, verbose=3)
    grid = GridSearchCV(SVC(), param_grid, n_jobs=3, verbose=3)

    # fitting the model for grid search
    grid.fit(X_train, y_train)

    results = [grid]

    # summarize result
    print('Best Score: %s' % grid.best_score_)
    print('Best Hyperparameters: %s' % grid.best_params_)

    # Print the test score of the best model
    clfRFC = grid.best_estimator_
    print('Test accuracy: %.3f' % clfRFC.score(X_test, y_test))

    # who knows?
    grid_predictions = grid.predict(X_test)
    print(classification_report(y_test, grid_predictions))

    results.append(grid_predictions)
    elem = [X_train, X_test, y_train, y_test, side]
    results.append(elem)
    pickle_out = open(path + 'results' + side + 'SVM'+str(i)+'.pickle', 'wb')
    pickle.dump(results, pickle_out)
    pickle_out.close()


if __name__ == '__main__':
    poseModel = analyzeModel('BODY_25')
    sx2dx = '/sx2dx/'
    dx2sx = '/dx2sx/'
    both = '/both/'
    # direction = ['sx2dx', 'dx2sx', 'both']
    directions = [sx2dx, dx2sx, both]
    pathDir = 'C:/Users/raimo/Desktop/lastPythonProject/'
    checkDir(pathDir+'/results')
    for side in directions:
        checkDir(pathDir+'/results' + side)
        i = 0
        path2CsvFiles = unpicklingData(FINAL_SVM_CSV_PATH, poseModel + side)
        X_train, X_test, y_train, y_test = None, None, None, None
        for path in path2CsvFiles:
            df = pd.read_csv(path, index_col=0)
            print(f'number of rows/examples and columns in the dataset: {df.shape}')
            print(path)
            print(df.head())
            X = df.drop('Target', axis=1)
            y = df['Target']

            X = np.nan_to_num(X)
            sc = StandardScaler()
            X_std = sc.fit_transform(X)

            pca = PCA()
            pca.fit(X_std)

            X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.20, random_state=42)

            # logisticRegressionGrid(X_train, X_test, y_train, y_test, side, i)
            # decisionTreeGrid(X_train, X_test, y_train, y_test, side, i)
            randomForestGrid(X_train, X_test, y_train, y_test, side, i)
            # svmGrid(X_train, X_test, y_train, y_test, side, i)
            i = i+1
