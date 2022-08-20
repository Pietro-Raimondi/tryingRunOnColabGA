import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from dirManager import analyzeModel, unpicklingData, FINAL_SVM_CSV_PATH


# def randomForestGrid(path2CsvFiles):
#     # df = pd.read_csv(path2CsvFiles[0])
#     df = pd.read_csv(path2CsvFiles, index_col=0)
#     print(f'number of rows/examples and columns in the dataset: {df.shape}')
#     print(path2CsvFiles)
#     X = df.drop('Target', axis=1)
#
#     X = np.nan_to_num(X)
#     sc = StandardScaler()
#     sc.fit(X)
#     X = sc.transform(X)
#
#     y = df['Target']
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
#
#     pipelineRFC = make_pipeline(StandardScaler(), RandomForestClassifier(criterion='gini', random_state=1))
#     #
#     # Create the parameter grid
#     #
#
#     '''
#     param_grid_rfc = [{
#         'randomforestclassifier__max_depth': [50],
#         'randomforestclassifier__max_features': [int(len(X[0]))]
#     }]'''
#
#     param_grid_rfc = [{
#         'randomforestclassifier__max_depth': [25, 50, 75, 100],
#         'randomforestclassifier__max_features': [int(len(X[0])), int(np.sqrt(len(X[0]))), int(len(X[0])/2), int(len(X[0])/3)]
#     }]
#
#     #
#     # Create an instance of GridSearch Cross-validation estimator
#     #
#     gsRFC = GridSearchCV(estimator=pipelineRFC,
#                          param_grid=param_grid_rfc,
#                          scoring='accuracy',
#                          cv=10,
#                          refit=True,
#                          n_jobs=1)
#     #
#     # Train the RandomForestClassifier
#     #
#     gsRFC = gsRFC.fit(X_train, y_train)
#     #
#     # Print the training score of the best model
#     #
#     print(gsRFC.best_score_)
#     #
#     # Print the model parameters of the best model
#     #
#     print(gsRFC.best_params_)
#     #
#     # Print the test score of the best model
#     #
#     clfRFC = gsRFC.best_estimator_
#     print('Test accuracy: %.3f' % clfRFC.score(X_test, y_test))


def logisticRegressionGrid(X_train, X_test, y_train, y_test):
    model = LogisticRegression()
    # define evaluation
    # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # define search space
    space = dict()
    space['solver'] = ['liblinear', 'saga']
    # space['solver'] = ['newton-cg', 'lbfgs', 'liblinear', 'svd', 'cholesky', 'lsqr', 'sag']
    space['penalty'] = ['l2', 'elasticnet']
    space['C'] = [1e-5, 1e-1, 1, 100]
    space['max_iter'] = [20000]
    '''    
    space['solver'] = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    # space['solver'] = ['newton-cg', 'lbfgs', 'liblinear', 'svd', 'cholesky', 'lsqr', 'sag']
    space['penalty'] = ['l1', 'l2', 'elasticnet']
    space['C'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 50, 100]
    space['max_iter'] = [20000]'''
    # define search
    # search = GridSearchCV(model, space, scoring='accuracy', n_jobs=-1, cv=5)
    search = GridSearchCV(model, space, scoring='accuracy', cv=3)

    # execute search
    result = search.fit(X_train, y_train)

    # summarize result
    print('Best Score: %s' % result.best_score_)
    print('Best Hyperparameters: %s' % result.best_params_)

    # Print the test score of the best model
    clfRFC = result.best_estimator_
    print('Test accuracy: %.3f' % clfRFC.score(X_test, y_test))

    # who knows?
    grid_predictions = result.predict(X_test)
    print(classification_report(y_test, grid_predictions))


def decisionTreeGrid(X_train, X_test, y_train, y_test):
    param_grid = {'max_features': ['auto', 'sqrt', 'log2'],
                  'ccp_alpha': [0.1, .01, .001],
                  'max_depth': [5, 6, 7, 8, 9, None],
                  'criterion': ['gini', 'entropy']
                  }
    tree_clas = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=tree_clas, param_grid=param_grid, cv=5, verbose=True)
    grid_search.fit(X_train, y_train)

    # Print the training score of the best model
    print(grid_search.best_score_)

    # Print the model parameters of the best model
    print(grid_search.best_params_)

    # Print the test score of the best model
    clfRFC = grid_search.best_estimator_
    print('Test accuracy: %.3f' % clfRFC.score(X_test, y_test))

    # who knows?
    grid_predictions = grid_search.predict(X_test)
    print(classification_report(y_test, grid_predictions))


def randomForestGrid(X_train, X_test, y_train, y_test):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt', 'log2']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    param_grid = {'criterion': ['gini', 'entropy'],
                   'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rfc = RandomForestClassifier(random_state=42)
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, verbose=3)
    CV_rfc.fit(X_train, y_train)

    # Print the training score of the best model
    print(CV_rfc.best_score_)

    # Print the model parameters of the best model
    print(CV_rfc.best_params_)

    # Print the test score of the best model
    clfRFC = CV_rfc.best_estimator_
    print('Test accuracy: %.3f' % clfRFC.score(X_test, y_test))

    # who knows?
    grid_predictions = CV_rfc.predict(X_test)
    print(classification_report(y_test, grid_predictions))


def svmGrid(X_train, X_test, y_train, y_test):
    # defining parameter range
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf', 'linear']}

    # grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
    grid = GridSearchCV(SVC(), param_grid, verbose=3)

    # fitting the model for grid search
    grid.fit(X_train, y_train)

    # print best parameter after tuning
    print(grid.best_params_)

    # print how our model looks after hyper-parameter tuning
    print(grid.best_estimator_)

    grid_predictions = grid.predict(X_test)

    # print classification report
    print(classification_report(y_test, grid_predictions))


if __name__ == '__main__':
    poseModel = analyzeModel('BODY_25')
    sx2dx = '/sx2dx/'
    dx2sx = '/dx2sx/'
    both = '/both/'
    path2CsvFiles = unpicklingData(FINAL_SVM_CSV_PATH, poseModel + dx2sx)
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

        # logisticRegressionGrid(X_train, X_test, y_train, y_test)
        # decisionTreeGrid(X_train, X_test, y_train, y_test)
        # randomForestGrid(X_train, X_test, y_train, y_test)
        svmGrid(X_train, X_test, y_train, y_test)
