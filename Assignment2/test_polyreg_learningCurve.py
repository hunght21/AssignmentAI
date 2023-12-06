'''
    TEST SCRIPT FOR POLYNOMIAL REGRESSION 1
    AUTHOR Eric Eaton, Xiaoxiang Hu
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from polyreg import learningCurve

#----------------------------------------------------
# Plotting tools

def plotLearningCurve(errorTrain, errorTest, regLambda, degree, ax):
    '''
        plot computed learning curve
    '''
    minX = 3
    maxY = max(errorTest[minX+1:])

    xs = np.arange(len(errorTrain))
    ax.plot(xs, errorTrain, 'r-o')
    ax.plot(xs, errorTest, 'b-o')
    ax.plot(xs, np.ones(len(xs)), 'k--')
    ax.legend(['Training Error', 'Testing Error'], loc='best')
    ax.set_title('Learning Curve (d='+str(degree)+', lambda='+str(regLambda)+')')
    ax.set_xlabel('Training samples')
    ax.set_ylabel('Error')
    ax.set_yscale('log')
    ax.set_ylim((0, maxY))
    ax.set_xlim((minX, 10))

def generateLearningCurve(X, y, degree, regLambda, ax):
    '''
        computing learning curve via leave one out CV
    '''

    n = len(X)

    errorTrains = np.zeros((n, n-1))
    errorTests = np.zeros((n, n-1))

    for itrial in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, random_state=itrial)

        (errTrain, errTest) = learningCurve(X_train, y_train, X_test, y_test, regLambda, degree)

        errorTrains[itrial, :] = errTrain
        errorTests[itrial, :] = errTest

    errorTrain = errorTrains.mean(axis=0)
    errorTest = errorTests.mean(axis=0)

    plotLearningCurve(errorTrain, errorTest, regLambda, degree, ax)

#-----------------------------------------------

if __name__ == "__main__":
    '''
        Main function to test polynomial regression
    '''

    # load the data
    filePath = r"C:\Users\ACER\OneDrive - Hanoi University of Science and Technology\Documents\Desktop\AI-2023\AssAI\Assignment2\polydata.dat"
    file = open(filePath, 'r')
    allData = np.loadtxt(file, delimiter=',')

    X = allData[:, 0]
    y = allData[:, 1]

    # generate Learning curves for different params
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    generateLearningCurve(X, y, 1, 0, axes[0, 0])
    generateLearningCurve(X, y, 4, 0, axes[0, 1])
    generateLearningCurve(X, y, 8, 0, axes[0, 2])
    generateLearningCurve(X, y, 8, 0.1, axes[1, 0])
    generateLearningCurve(X, y, 8, 1, axes[1, 1])
    generateLearningCurve(X, y, 8, 100, axes[1, 2])

    plt.tight_layout()
    plt.show()
