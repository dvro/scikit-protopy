# -*- coding: utf-8 -*-
"""
All K-Nearest Neighbors
"""

# Author: Dayvid Victor <victor.dvro@gmail.com>
#
# License: BSD 3 clause

import numpy as np


from sklearn.utils.validation import check_X_y

from ..base import InstanceReductionMixin
from protopy.selection.enn import ENN


class AllKNN(InstanceReductionMixin):
    """All K-Nearest Neighbors.

    The All KNN removes the instances in the boundaries, maintaining 
    redudant samples. Creating a much more smooth decision region.
    It is similar to the Repeated-Edited Nearest Neighbors, but it has
    a different approach.

    Parameters
    ----------
    n_neighbors : int, optional (default = 5)
        Number of limit neighbors to use by default for :meth:`k_neighbors` queries.

    Attributes
    ----------
    `X_` : array-like, shape = [indeterminated, n_features]
        Selected prototypes.

    `y_` : array-like, shape = [indeterminated]
        Labels of the selected prototypes.

    `reduction_` : float, percentual of reduction.

    Examples
    --------
    >>> from protopy.selection.allknn import AllKNN
    >>> import numpy as np
    >>> X = np.array([[-1, 0], [-0.8, 1], [-0.8, -1], [-0.5, 0] , [0.5, 0], [1, 0], [0.8, 1], [0.8, -1]])
    >>> y = np.array([1, 1, 1, 2, 1, 2, 2, 2])
    >>> all_kneigh = AllKNN()
    >>> all_kneigh.fit(X, y)
    AllKNN(n_neighbors=3)
    >>> print(all_kneigh.predict([[-0.6, 0.6]]))
    [1]
    >>> print all_kneigh.reduction_
    0.625

    See also
    --------
    protopy.selection.enn.ENN: edited nearest neighbor
    protopy.selection.renn.RENN: repeated edited nearest neighbor

    References
    ----------
    I. Tomek. An experiment with the edited nearest-neighbor rule. 
    IEEE Transactions on Systems, Man, and Cybernetics, 6(6):448–452, 1976.

    """

    def __init__(self, n_neighbors= 5):
        self.n_neighbors = n_neighbors
        self.classifier = None

    def reduce_data(self, X, y):
        X, y = check_X_y(X, y, accept_sparse="csr")

        classes = np.unique(y)
        self.classes_ = classes

        edited_nn = ENN(n_neighbors = 1)
        p_, l_, r_ = X, y, 1.0

        for k in range(1, self.n_neighbors + 1):
            if l_.shape[0] > k + 1:
                edited_nn.n_neighbors = k
                edited_nn.fit(p_, l_)
                p_ = edited_nn.X_
                l_ = edited_nn.y_
                r_ = edited_nn.reduction_
             
        self.X_ = p_
        self.y_ = l_
        self.reduction_ = 1.0 - float(l_.shape[0]) / y.shape[0]

        return self.X_, self.y_
   
