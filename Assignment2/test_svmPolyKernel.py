# test_svmPolyKernel.py

"""
Test SVM with custom polynomial kernels
"""

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from svmKernels import MyModifiedPolynomialKernel

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data
Y = iris.target

print("Training the SVMs...")

C = 1.0  # value of C for the SVMs

# Create an instance of SVM with the modified custom polynomial kernel and train it
myModel = svm.SVC(C=C, kernel=MyModifiedPolynomialKernel.compute)
myModel.fit(X, Y)

# Create an instance of SVM with the built-in poly kernel and train it
model = svm.SVC(C=C, kernel='poly', degree=3, coef0=1, gamma=1)
model.fit(X, Y)

print("Testing the SVMs...")

h = .02  # step size in the mesh

# Plot the decision boundary for the first two features
feature1_index = 0
feature2_index = 1

x_min, x_max = X[:, feature1_index].min() - 1, X[:, feature1_index].max() + 1
y_min, y_max = X[:, feature2_index].min() - 1, X[:, feature2_index].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Get predictions for both custom model and built-in model
myPredictions = myModel.predict(np.c_[xx.ravel(), yy.ravel()])
myPredictions = myPredictions.reshape(xx.shape)

predictions = model.predict(np.c_[xx.ravel(), yy.ravel()])
predictions = predictions.reshape(xx.shape)

# Plot custom model results
plt.subplot(1, 2, 1)
plt.pcolormesh(xx, yy, myPredictions, cmap=plt.cm.Paired)
plt.scatter(X[:, feature1_index], X[:, feature2_index], c=Y, cmap=plt.cm.Paired)
plt.title("SVM with My Custom Polynomial Kernel (degree = 3, C = " + str(C) + ")")
plt.xlabel('Feature ' + str(feature1_index + 1))
plt.ylabel('Feature ' + str(feature2_index + 1))
plt.axis('tight')

# Plot built-in model results
plt.subplot(1, 2, 2)
plt.pcolormesh(xx, yy, predictions, cmap=plt.cm.Paired)
plt.scatter(X[:, feature1_index], X[:, feature2_index], c=Y, cmap=plt.cm.Paired)
plt.title("SVM with Built-in Polynomial Kernel (degree = 3, C = " + str(C) + ")")
plt.xlabel('Feature ' + str(feature1_index + 1))
plt.ylabel('Feature ' + str(feature2_index + 1))
plt.axis('tight')

plt.show()
