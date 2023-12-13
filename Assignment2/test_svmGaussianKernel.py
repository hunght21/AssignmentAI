# svmGaussianKernel.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from svmKernels import myGaussianKernel

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features
Y = iris.target

print("Training the SVMs...")

C = 1.0  # value of C for the SVMs

# create an instance of SVM with the custom kernel and train it
gaussSigma = 10  # Set sigma for the Gaussian kernel
gamma_value = 1.0 / (2 * gaussSigma ** 2)  # Compute gamma for the SVM

myModel = svm.SVC(C=C, kernel=myGaussianKernel, gamma=gamma_value)
myModel.fit(X, Y)

print("")
print("Testing the SVMs...")

h = .02  # step size in the mesh

# Plot the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# get predictions for the custom model
myPredictions = myModel.predict(np.c_[xx.ravel(), yy.ravel()])
myPredictions = myPredictions.reshape(xx.shape)

# Plotting the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.pcolormesh(xx, yy, myPredictions, cmap=plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
plt.title("SVM with Custom Gaussian Kernel (sigma = " + str(gaussSigma) + ", C = " + str(C) + ")")
plt.axis('tight')

# For comparison, using the built-in RBF kernel
equivalentGamma = 1.0 / (2 * gaussSigma ** 2)
model = svm.SVC(C=C, kernel='rbf', gamma=equivalentGamma)
model.fit(X, Y)

predictions = model.predict(np.c_[xx.ravel(), yy.ravel()])
predictions = predictions.reshape(xx.shape)

plt.subplot(1, 2, 2)
plt.pcolormesh(xx, yy, predictions, cmap=plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
plt.title('SVM with Equivalent Scikit-learn RBF Kernel for Comparison')
plt.axis('tight')

plt.show()

# Describe the effects of varying parameters in the README file
# SVM_KERNELPARAMETEREFFECTS:
# As C increased, I observed that...
# Describe the behavior of the SVM as C and sigma vary.
