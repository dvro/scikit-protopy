import numpy as np
import matplotlib.pyplot as plt

from protopy.selection.enn import ENN
from protopy.selection.cnn import CNN
from protopy.selection.renn import RENN
from protopy.selection.allknn import AllKNN
from protopy.selection.tomek_links import TomekLinks

f, subfig = plt.subplots(3,2)



mu1 = [4, 5]
si1 = [[0.75, 0.25], [0.25, 0.75]]

mu2 = [5, 5]
si2 = [[0.25, 0.75], [0.75, 0.25]]

samples = 100

X1 = np.random.multivariate_normal(
    np.asarray(mu1), np.asarray(si1), samples)
X2 = np.random.multivariate_normal(
    np.asarray(mu2), np.asarray(si2), samples)
X = np.vstack((X1, X2))
y = np.asarray([0] * samples + [1] * samples)


algorithms = [ENN(), RENN(), AllKNN(), TomekLinks(), CNN()]
titles = ['ENN','RENN', 'AllKNN', 'TomekLinks', 'CNN' ]
index = 0

for i in range(3):
    for j in range(2):
        if i == 0 and j == 0:
            subfig[i][j].plot(X[y==0].T[0], X[y==0].T[1], 'bs', X[y==1].T[0], X[y==1].T[1],'ro')
            subfig[i][j].axis([0, 10, 0, 10])
            subfig[i][j].set_title('Original Dataset')
        elif index < len(algorithms):
            X_, y_ = algorithms[index].reduce_data(X, y)
            subfig[i][j].plot(X_[y_==0].T[0], X_[y_==0].T[1], 'bs', X_[y_==1].T[0], X_[y_==1].T[1],'ro')
            subfig[i][j].axis([0, 10, 0, 10])
            subfig[i][j].set_title(titles[index])
            index = index + 1

plt.show()
