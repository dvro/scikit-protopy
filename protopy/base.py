"""Base and mixin classes for instance reduction techniques"""
# Author: Dayvid Victor <dvro@cin.ufpe.br>
# License: BSD Style
import warnings
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.base import BaseEstimator
from sklearn.neighbors.classification import KNeighborsClassifier

from sklearn.utils import safe_asarray, atleast2d_or_csr, check_arrays
from sklearn.externals import six

class InstanceReductionWarning(UserWarning):
    pass

# Make sure that NeighborsWarning are displayed more than once
warnings.simplefilter("always", InstanceReductionWarning)


class InstanceReductionBase(six.with_metaclass(ABCMeta, BaseEstimator)):
    """Base class for instance reduction estimators."""

    @abstractmethod
    def __init__(self):
        pass


class InstanceReductionMixin(InstanceReductionBase):
    """Mixin class for all instance reduction techniques"""

    def reduce_data(self, X, y):
        """Perform the instance reduction procedure on the given training data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training set.0

        y : array-like, shape = [n_samples]
            Labels for X.

        Returns
        -------
        P : array-like, shape = [indeterminated, n_features]
            Resulting training set.
        
        q : array-like, shape = [indertaminated]
            Labels for P
        """
        pass


    def fit(self, X, y, reduce_data=True):
        """
        Fit the InstanceReduction model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
            Note that centroid shrinking cannot be used with sparse matrices.
        y : array, shape = [n_samples]
            Target values (integers)
	reduce_data : bool, flag indicating if the reduction would be performed
        """
	self.X = X
	self.y = y
	
	if reduce_data:
		self.reduce(X, y)

        return self

    def predict(self, X, n_neighbors = 1):
        """Perform classification on an array of test vectors X.

        The predicted class C for each sample in X is returned.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array, shape = [n_samples]

        Notes
        -----
	The default prediction is using KNeighborsClassifier, if the
	instance reducition algorithm is to be performed with another
	classifier, it should be explicited overwritten and explained
	in the documentation.
        """
        X = atleast2d_or_csr(X)
        if not hasattr(self, "prototypes_") or self.prototypes_ == None:
            raise AttributeError("Model has not been trained yet.")
        #return self.labels_[pairwise_distances(
        #    X, self.prototypes_, metric=self.metric).argmin(axis=1)]
        knn = KNeighborsClassifier(n_neighbors = n_neighbors)
        knn.fit(self.prototypes_, self.labels_)
        return knn.predict(X)


