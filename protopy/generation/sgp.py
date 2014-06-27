# -*- coding: utf-8 -*-
"""
Self-Generating Prototypes
"""

# Author: Dayvid Victor <victor.dvro@gmail.com>
#
# License: BSD 3 clause

import numpy as np
from scipy import sparse as sp

from sklearn.externals.six.moves import xrange
from sklearn.utils.validation import check_arrays, atleast2d_or_csr
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.decomposition import PCA

from ..base import InstanceReductionMixin

class _Group(object):

    def __init__(self, X, label):
        self.X = X
        self.label = label
        if len(X) > 0:
            self.update_all()

    def __add__(self, other):
        X = np.vstack((self.X, other.X))
        return _Group(X, self.label, update=True)

    def __len__(self):
        length = self.X.shape[0] if self.X != None else 0
        return length

    def add_instances(self, X, update=False):
        self.X = np.vstack((self.X,X))
        if update:
            self.update_all()

    def remove_instances(self, indexes, update=False):
        _X = self.X[indexes]
        self.X = np.delete(self.X, indexes, axis=0)
        if update:
            self.update_all()
        return _X

    def update_all(self):
        self.rep_x = np.mean(self.X, axis=0)

class SGP(InstanceReductionMixin):
    """Self-Generating Prototypes

    The Self-Generating Prototypes generates instances is a centroid-based
    prototype generation algorithm that uses the space spliting mechanism
    to generate prototypes in the center of each cluster.

    Parameters
    ----------
    r_min: float, optional (default = 0.0)
        Determine the minimum size of a cluster [0.00, 0.20]

    r_mis: float, optional (default = 0.0)
        Determine the error tolerance before split a group


    Attributes
    ----------
    `X_` : array-like, shape = [indeterminated, n_features]
        Selected prototypes.

    `y_` : array-like, shape = [indeterminated]
        Labels of the selected prototypes.

    `reduction_` : float, percentual of reduction.

    Examples
    --------
    >>> from protopy.generation.sgp import SGP
    >>> import numpy as np
    >>> X = np.array([[i] for i in range(1,13)])
    >>> X = X + np.asarray([0.1,0,-0.1,0.1,0,-0.1,0.1,-0.1,0.1,-0.1,0.1,-0.1])
    >>> y = np.array([1, 1, 1, 2, 2, 2, 1, 1, 2, 2, 1, 1])
    >>> sgp = SGP()
    >>> sgp.fit(X, y)
    SGP(r_min=0.0, r_mis=0.0)
    >>> print sgp.predict(X)
    [1 1 1 2 2 2 1 1 2 2 1 1]
    >>> print sgp.reduction_
    0.5

    See also
    --------
    protopy.generation.sgp.SGP2: self-generating prototypes 2

    References
    ----------
    Hatem A. Fayed, Sherif R Hashem, and Amir F Atiya. Self-generating prototypes
    for pattern classification. Pattern Recognition, 40(5):1498–1509, 2007.
    """

    def __init__(self, r_min=0.0, r_mis=0.0):
        self.groups = None
        self.r_min = r_min
        self.r_mis = r_mis
        self.n_neighbors = 1
        self.classifier = None
        self.groups = None


    def reduce_data(self, X, y):
        X, y = check_arrays(X, y, sparse_format="csr")

        if self.classifier == None:
            self.classifier = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        if self.classifier.n_neighbors != self.n_neighbors:
            self.classifier.n_neighbors = self.n_neighbors

        classes = np.unique(y)
        self.classes_ = classes

        # loading inicial groups
        self.groups = []
        for label in classes:
            mask = y == label
            self.groups = self.groups + [_Group(X[mask], label)]

        self.__main_loop()
        self.__generalization_step()

        self.X_ = np.asarray([g.rep_x for g in self.groups])
        self.y_ = np.asarray([g.label for g in self.groups])
        self.reduction_ = 1.0 - float(len(self.y_))/len(y)

        return self.X_, self.y_
     
    def __main_loop(self):
        exit_count = 0
        knn = KNeighborsClassifier(n_neighbors = 1, algorithm='brute')
        while exit_count < len(self.groups):
            index, exit_count = 0, 0
            while index < len(self.groups):

                group = self.groups[index]
                reps_x = np.asarray([g.rep_x for g in self.groups])
                reps_y = np.asarray([g.label for g in self.groups])
                knn.fit(reps_x, reps_y)
                
                nn_idx = knn.kneighbors(group.X, n_neighbors=1, return_distance=False)
                nn_idx = nn_idx.T[0]
                mask = nn_idx == index
                
                # if all are correctly classified
                if not (False in mask):
                    exit_count = exit_count + 1
                
                # if all are misclasified
                elif not (group.label in reps_y[nn_idx]):
                    pca = PCA(n_components=1)
                    pca.fit(group.X)
                    # maybe use a 'for' instead of creating array
                    d = pca.transform(reps_x[index])
                    dis = [pca.transform(inst)[0] for inst in group.X]
                    mask_split = (dis < d).flatten()
                    
                    new_X = group.X[mask_split]
                    self.groups.append(_Group(new_X, group.label))
                    group.X = group.X[~mask_split]
                
                elif (reps_y[nn_idx] == group.label).all() and (nn_idx != index).any():
                    mask_mv = nn_idx != index
                    index_mv = np.asarray(range(len(group)))[mask_mv]
                    X_mv = group.remove_instances(index_mv)
                    G_mv = nn_idx[mask_mv]                        

                    for x, g in zip(X_mv, G_mv):
                        self.groups[g].add_instances([x])

                elif (reps_y[nn_idx] != group.label).sum()/float(len(group)) > self.r_mis:
                    mask_mv = reps_y[nn_idx] != group.label
                    new_X = group.X[mask_mv]
                    self.groups.append(_Group(new_X, group.label))
                    group.X = group.X[~mask_mv]
                else:
                   exit_count = exit_count + 1

                if len(group) == 0:
                    self.groups.remove(group)
                else:
                    index = index + 1

                for g in self.groups:
                    g.update_all()

        return self.groups                     


    def __generalization_step(self):
        larger = max([len(g) for g in self.groups])
        for group in self.groups:
            if len(group) < self.r_min * larger:
                self.groups.remove(group)
        return self.groups



