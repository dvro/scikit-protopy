# -*- coding: utf-8 -*-
"""
Tomek Links
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


class TomekLinks(InstanceReductionMixin):

    """Tomek Links.

    The Tomek Links algorithm removes a pair instances that
    forms a Tomek Link. This techniques removes instances in
    the decision region.

    Parameters
    ----------
    n_neighbors : int, optional (default = 3)
        Number of neighbors to use by default in the classification (only).
        The Tomek Links uses only n_neighbors=1 in the reduction.

    keep_class : int, optional (default = None)
        Label of the class to not be removed in the tomek links. If None,
        it removes all nodes of the links.

    Attributes
    ----------
    `X_` : array-like, shape = [indeterminated, n_features]
        Selected prototypes.

    `y_` : array-like, shape = [indeterminated]
        Labels of the selected prototypes.

    `reduction_` : float, percentual of reduction.

    Examples
    --------

    >>> from protopy.selection.tomek_links import TomekLinks
    >>> import numpy as np
    >>> X = np.array([[0],[1],[2.1],[2.9],[4],[5],[6],[7.1],[7.9],[9]])
    >>> y = np.array([1,1,2,1,2,2,2,1,2,2])
    >>> tl = TomekLinks()
    >>> tl.fit(X, y)
    TomekLinks(keep_class=None)
    >>> print tl.predict([[2.5],[7.5]])
    [1, 2]
    >>> print tl.reduction_
    0.4

    See also
    --------
    protopy.selection.enn.ENN: edited nearest neighbor

    References
    ----------
    I. Tomek, “Two modifications of cnn,” IEEE Transactions on Systems,
    Man and Cybernetics, vol. SMC-6, pp. 769–772, 1976.

    """

    def __init__(self, n_neighbors=3, keep_class=None):
        self.n_neighbors = n_neighbors
        self.classifier = None
        self.keep_class = keep_class


    def reduce_data(self, X, y):
        if self.classifier == None:
            self.classifier = KNeighborsClassifier(n_neighbors=self.n_neighbors, algorithm='brute')
        if self.classifier.n_neighbors != self.n_neighbors:
            self.classifier.n_neighbors = self.n_neighbors

        X, y = check_arrays(X, y, sparse_format="csr")

        classes = np.unique(y)
        self.classes_ = classes
        self.classifier.fit(X, y)
        nn_idx = self.classifier.kneighbors(X, n_neighbors=2, return_distance=False)
        nn_idx = nn_idx.T[1]

        mask = [nn_idx[nn_idx[index]] == index and y[index] != y[nn_idx[index]] for index in xrange(nn_idx.shape[0])]
        mask = ~np.asarray(mask) 
        if self.keep_class != None and self.keep_class in self.classes_:
            mask[y==self.keep_class] = True

        self.X_ = np.asarray(X[mask])
        self.y_ = np.asarray(y[mask])
        self.reduction_ = 1.0 - float(len(self.y_)) / len(y)

        return self.X_, self.y_

