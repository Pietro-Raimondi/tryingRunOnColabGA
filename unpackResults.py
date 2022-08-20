import pickle
from sklearn.metrics import classification_report


def printingResults(side, classificationType):
    path = 'C:/Users/raimo/Desktop/lastPythonProject/results/'
    for i in range(4):
        # stri = '/' + direction[0]+'RegressionGrid'+str(i)
        pickle_in = open(path + side + classificationType + str(i) + '.pickle', 'rb')
        print(side + classificationType + str(i))
        data = pickle.load(pickle_in)

        # summarize result
        print('Best Score: %s' % data[0].best_score_)
        print('Best Hyperparameters: %s' % data[0].best_params_)

        # Print the test score of the best model
        clfRFC = data[0].best_estimator_
        print('Test accuracy: %.3f' % clfRFC.score(data[2][1], data[2][3]))

        print(classification_report(data[2][3], data[1]))


if __name__ == '__main__':
    sx2dx = '/sx2dx/'
    dx2sx = '/dx2sx/'
    both = '/both/'
    directions = [sx2dx, dx2sx, both]
    classificationType = ['RegressionGrid', 'DecisionTree', 'RandomForest', 'SVM']
    '''    
    for type in classificationType:
        for side in directions:
            printingResults(side, type)
    '''
    for side in directions:
        printingResults(side, classificationType[3])
