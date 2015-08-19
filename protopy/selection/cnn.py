# -*- coding: utf-8 -*-
"""
Condensed-Nearest Neighbors
"""

# Author: Dayvid Victor <victor.dvro@gmail.com>
#
# License: BSD 3 clause

import numpy as np


from sklearn.utils.validation import check_X_y
from sklearn.neighbors.classification import KNeighborsClassifier

from ..base import InstanceReductionMixin


class CNN(InstanceReductionMixin):
    """Condensed Nearest Neighbors.

    Each class is represented by a set of prototypes, with test samples
    classified to the class with the nearest prototype.
    The Condensed Nearest Neighbors removes the redundant instances,
    maintaining the samples in the decision boundaries.

    Parameters
    ----------
    n_neighbors : int, optional (default = 1)
        Number of neighbors to use by default for :meth:`k_neighbors` queries.

    Attributes
    ----------
    `prototypes_` : array-like, shape = [indeterminated, n_features]
        Selected prototypes.

    `labels_` : array-like, shape = [indeterminated]
        Labels of the selected prototypes.

    `reduction_` : float, percentual of reduction.

    Examples
    --------
    >>> from protopy.selection.cnn import CNN
    >>> import numpy as np
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> y = np.array([1, 1, 1, 2, 2, 2])
    >>> cnn = CNN()
    >>> cnn.fit(X, y)
    CNN(n_neighbors=1)
    >>> print(cnn.predict([[-0.8, -1]]))
    [1]

    See also
    --------
    sklearn.neighbors.KNeighborsClassifier: nearest neighbors classifier

    Notes
    -----
    The Condensed Nearest Neighbor is one the first prototype selection
    technique in literature.

    References
    ----------
    P. E. Hart, The condensed nearest neighbor rule, IEEE Transactions on 
    Information Theory 14 (1968) 515–516.

    """

    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors
        self.classifier = None

    def reduce_data(self, X, y):
        
        X, y = check_X_y(X, y, accept_sparse="csr")

        if self.classifier == None:
            self.classifier = KNeighborsClassifier(n_neighbors=self.n_neighbors)

        prots_s = []
        labels_s = []

        classes = np.unique(y)
        self.classes_ = classes

        for cur_class in classes:
            mask = y == cur_class
            insts = X[mask]
            prots_s = prots_s + [insts[np.random.randint(0, insts.shape[0])]]
            labels_s = labels_s + [cur_class]


        self.classifier.fit(prots_s, labels_s)
        for sample, label in zip(X, y):
            if self.classifier.predict(sample) != [label]:
                prots_s = prots_s + [sample]
                labels_s = labels_s + [label]
                self.classifier.fit(prots_s, labels_s)
       
        self.X_ = np.asarray(prots_s)
        self.y_ = np.asarray(labels_s)
        self.reduction_ = 1.0 - float(len(self.y_))/len(y)
        return self.X_, self.y_
 
