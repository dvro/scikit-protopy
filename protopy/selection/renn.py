# -*- coding: utf-8 -*-
"""
Repeated Edited-Nearest Neighbors
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

from .enn import ENN

class RENN(InstanceReductionMixin):
    """Repeated Edited Nearest Neighbors.

    The Repeated Edited Nearest Neighbors  removes the instances in the 
    boundaries, maintaining redudant samples. Creating a much more smooth
    decision region.

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
    >>> from protopy.selection.renn import RENN
    >>> import numpy as np
    >>> X = np.array([[-1, 0], [-0.8, 1], [-0.8, -1], [-0.5, 0] , [0.5, 0], [1, 0], [0.8, 1], [0.8, -1]])
    >>> y = np.array([1, 1, 1, 2, 1, 2, 2, 2])
    >>> repeated_enn = RENN()
    >>> repeated_enn.fit(X, y)
    RENN(n_neighbors=3)
    >>> print(repeated_enn.predict([[-0.6, 0.6]]))
    [1]
    >>> print repeated_enn.reduction_
    0.25

    See also
    --------
    protopy.selection.enn.ENN: edited nearest neighbor

    References
    ----------
    Dennis L. Wilson. Asymptotic properties of nearest neighbor rules 
    using edited data. Systems, Man and Cybernetics, IEEE Transactions
    on, 2(3):408–421, July 1972.
    """

    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
        self.classifier = None

    def reduce_data(self, X, y):
        X, y = check_arrays(X, y, sparse_format="csr")

        classes = np.unique(y)
        self.classes_ = classes

        enn = ENN(n_neighbors = self.n_neighbors)
        p_, l_, r_ = X, y, 1.0

        while r_ != 0:
            enn.reduce_data(p_, l_)
            p_ = enn.X_
            l_ = enn.y_
            r_ = enn.reduction_
             
        self.X_ = p_
        self.y_ = l_
        self.reduction_ = 1.0 - float(l_.shape[0]) / y.shape[0]

        return self.X_, self.y_
 

