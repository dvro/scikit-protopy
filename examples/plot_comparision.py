#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
==============================================
Prototype Selection and Generation Comparision
==============================================
A comparison of a several prototype selection and generation algorithms in 
the project on synthetic datasets.
The point of this example is to illustrate the nature of decision boundaries
after applying instance reduction techniques.
This should be taken with a grain of salt, as the intuition conveyed by
these examples does not necessarily carry over to real datasets.

In particular in high dimensional spaces data can more easily be separated
linearly and the simplicity of classifiers such as naive Bayes and linear SVMs
might lead to better generalization.

The plots show training points in solid colors and testing points
semi-transparent. 

The lower right shows:
- S: score on the traning set.
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
from protopy.selection.enn import ENN
from protopy.selection.cnn import CNN
from protopy.selection.renn import RENN
from protopy.selection.allknn import AllKNN
from protopy.selection.tomek_links import TomekLinks
from protopy.selection.ssma import SSMA
from protopy.generation.sgp import SGP, SGP2, ASGP

h = .02  # step size in the mesh

names = ["KNN", "ENN", "CNN", "RENN", "AllKNN", "Tomek Links", "SGP", "SGP2", "ASGP", "SSMA"]


classifiers = [
#    KNeighborsClassifier(3),
    ENN(n_neighbors=3),
    CNN(n_neighbors=3),
#    RENN(n_neighbors=3),
#    AllKNN(n_neighbors=3),
#    TomekLinks(n_neighbors=1),
#    SGP(r_min=0.05, r_mis=0.05),
#    SGP2(r_min=0.05, r_mis=0.05),
    ASGP(r_min=0.05, r_mis=0.05),
    SSMA()]

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)

rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable
            ]

figure = pl.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds in datasets:
    # preprocess dataset, split into training and test part
    X, y = ds
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
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.2)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = pl.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
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

        if hasattr(clf, 'reduction_'):
            X_prot, y_prot = clf.X_, clf.y_
        else:
            X_prot, y_prot = X_train, y_train
        
        # Plot also the training points
        ax.scatter(X_prot[:, 0], X_prot[:, 1], c=y_prot, cmap=cm_bright)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   alpha=0.2)

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
