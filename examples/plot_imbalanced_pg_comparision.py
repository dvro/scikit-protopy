#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=======================================================
Prototype Generation Algorithms over Imbalanced Domains
=======================================================
A comparison of a several prototype generation algorithms over the project 
on synthetic imbalanced datasets.
The point of this example is to illustrate the nature of decision boundaries

The plots show training points in solid colors and testing points
semi-transparent. 

The lower right shows:
- S: score on the traning set (AUC)
- R: reduction ratio.

License: BSD 3 clause
"""

print(__doc__)


import numpy as np
import pylab as pl
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from protopy.selection.enn import ENN
from protopy.selection.cnn import CNN
from protopy.selection.renn import RENN
from protopy.selection.allknn import AllKNN
from protopy.selection.tomek_links import TomekLinks
from protopy.generation.sgp import SGP, SGP2, ASGP

h = .02  # step size in the mesh

names = ["KNN", "SGP", "SGP2", "ASGP"]


classifiers = [
    KNeighborsClassifier(3),
    SGP(r_min=0.2, r_mis=0.05),
    SGP2(r_min=0.2, r_mis=0.05),
    ASGP(r_min=0.2, r_mis=0.05)]

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)

rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable
            ]

def random_subset(iterator, k):
    result = iterator[:k]
    i = k
    tmp_it = iterator[k:]
    for item in tmp_it:
        i = i + 1
        s = int(np.random.random() * i)
        if s < k:
            result[s] = item
    return result

def generate_imbalance(X, y, positive_label=1, ir=2):
    mask = y == positive_label
    seq = np.arange(y.shape[0])[mask]
    k = float(sum(mask))/ir
    idx = np.asarray(random_subset(seq, int(k)))
    mask = ~mask
    mask[idx] = True
    return X[mask], y[mask]


figure = pl.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds in datasets:
    # preprocess dataset, split into training and test part
    X, y = ds
    X, y = generate_imbalance(X, y)

    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = pl.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = pl.subplot(len(datasets), len(classifiers) + 1, i)
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = pl.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        fp_rate, tp_rate, thresholds = roc_curve(
            y_test, y_pred, pos_label=1)
        score = auc(fp_rate, tp_rate)

        red = 0.0
        if  hasattr(clf, 'reduction_') and clf.reduction_ != None:
            red = clf.reduction_

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot also the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, 'S:' + ('%.2f' % score).lstrip('0') + '  R:' + ('%.2f' % red).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

figure.subplots_adjust(left=.02, right=.98)
pl.show()
