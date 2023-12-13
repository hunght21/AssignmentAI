"""
Custom SVM Kernels

Author: Eric Eaton, 2014

"""

import numpy as np


_polyDegree = 2
_gaussSigma = 1


class MyModifiedPolynomialKernel:
    @staticmethod
    def compute(X1, X2):
        """
        Compute the modified polynomial kernel between X1 and X2.

        Parameters:
        - X1: an n1-by-d numpy array of instances
        - X2: an n2-by-d numpy array of instances

        Returns:
        - An n1-by-n2 numpy array representing the Kernel (Gram) matrix
        """
        # Ensure X1 and X2 have the same number of dimensions
        if X1.ndim == 1:
            X1 = X1.reshape(1, -1)
        if X2.ndim == 1:
            X2 = X2.reshape(1, -1)

        return (np.dot(X1, X2.T) + 1) ** 2




def myGaussianKernel(X1, X2):
    '''
        Arguments:
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    n1,d1 = X1.shape
    n2,d2 = X2.shape

    # find the distance between the v and w vectors to be used in the 
    # K(v,w) equation
    distance = np.zeros((n1, n2))
    for i in range(0, n2):
      distance[:,i] = np.sum((X1 - X2[i,:]) ** 2, axis = 1)

    # compute the K(v,w) equation
    numerator = -distance
    denominator = 2 * (_gaussSigma ** 2)
    return np.exp(numerator / denominator)



def myCosineSimilarityKernel(X1,X2):
    '''
        Arguments:
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    return #TODO 
