# -*- coding: utf-8 -*-
"""
Edited-Nearest Neighbors
"""

# Author: Dayvid Victor <victor.dvro@gmail.com>
#
# License: BSD 3 clause

import numpy as np
from scipy import sparse as sp

from sklearn.externals.six.moves import xrange
from sklearn.utils.validation import check_arrays, atleast2d_or_csr
from sklearn.neighbors.classification import KNeighborsClassifier

from ..base import InstanceReductionMixin


class ENN(InstanceReductionMixin):

    """Edited Nearest Neighbors.

    The Edited Nearest Neighbors  removes the instances in de 
    boundaries, maintaining redudant samples.

    Parameters
    ----------
    n_neighbors : int, optional (default = 3)
        Number of neighbors to use by default for :meth:`k_neighbors` queries.

    Attributes
    ----------
    `X_` : array-like, shape = [indeterminated, n_features]
        Selected prototypes.

    `y_` : array-like, shape = [indeterminated]
        Labels of the selected prototypes.

    `reduction_` : float, percentual of reduction.

    Examples
    --------
    >>> from protopy.selection.enn import ENN
    >>> import numpy as np
    >>> X = np.array([[-1, 0], [-0.8, 1], [-0.8, -1], [-0.5, 0] , [0.5, 0], [1, 0], [0.8, 1], [0.8, -1]])
    >>> y = np.array([1, 1, 1, 2, 1, 2, 2, 2])
    >>> editednn = ENN()
    >>> editednn.fit(X, y)
    ENN(n_neighbors=3)
    >>> print(editednn.predict([[-0.6, 0.6]]))
    [1]
    >>> print editednn.reduction_
    0.25

    See also
    --------
    sklearn.neighbors.KNeighborsClassifier: nearest neighbors classifier

    References
    ----------
    Ruiqin Chang, Zheng Pei, and Chao Zhang. A modified editing k-nearest
    neighbor rule. JCP, 6(7):1493–1500, 2011.

    """

    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
        self.classifier = None


    def reduce_data(self, X, y):
        if self.classifier == None:
            self.classifier = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        if self.classifier.n_neighbors != self.n_neighbors:
            self.classifier.n_neighbors = self.n_neighbors

        X, y = check_arrays(X, y, sparse_format="csr")

        classes = np.unique(y)
        self.classes_ = classes

        mask = np.zeros(y.size, dtype=bool)

        tmp_m = np.ones(y.size, dtype=bool)
        for i in xrange(y.size):
            tmp_m[i] = not tmp_m[i]
            self.classifier.fit(X[tmp_m], y[tmp_m])
            sample, label = X[i], y[i]

            if self.classifier.predict(sample) == [label]:
                mask[i] = not mask[i]

            tmp_m[i] = not tmp_m[i]

        self.X_ = np.asarray(X[mask])
        self.y_ = np.asarray(y[mask])
        self.reduction_ = 1.0 - float(len(self.y_)) / len(y)
        return self.X_, self.y_


