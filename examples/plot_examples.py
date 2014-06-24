import numpy as np
import matplotlib.pyplot as plt

from protopy.selection.enn import ENN
from protopy.selection.cnn import CNN
from protopy.selection.renn import RENN

f, subfig = plt.subplots(4)



mu1 = [4, 5]
si1 = [[0.75, 0.25], [0.25, 0.75]]

mu2 = [6, 5]
si2 = [[0.25, 0.75], [0.75, 0.25]]

samples = 100

X1 = np.random.multivariate_normal(
    np.asarray(mu1), np.asarray(si1), samples)
X2 = np.random.multivariate_normal(
    np.asarray(mu2), np.asarray(si2), samples)
X = np.vstack((X1, X2))
y = np.asarray([0] * samples + [1] * samples)


subfig[0].plot(X[y==0].T[0], X[y==0].T[1], 'bs', X[y==1].T[0], X[y==1].T[1],'ro')
subfig[0].axis([0, 10, 0, 10])
subfig[0].set_title('Original Dataset')

editednn = ENN()
X_, y_ = editednn.reduce_data(X, y)
subfig[1].plot(X_[y_==0].T[0], X_[y_==0].T[1], 'bs', X_[y_==1].T[0], X_[y_==1].T[1],'ro')
subfig[1].axis([0, 10, 0, 10])
subfig[1].set_title('ENN')

condensednn = CNN()
X_, y_ = condensednn.reduce_data(X, y)
subfig[2].plot(X_[y_==0].T[0], X_[y_==0].T[1], 'bs', X_[y_==1].T[0], X_[y_==1].T[1],'ro')
subfig[2].axis([0, 10, 0, 10])
subfig[2].set_title('CNN')

renn = RENN()
X_, y_ = renn.reduce_data(X, y)
subfig[3].plot(X_[y_==0].T[0], X_[y_==0].T[1], 'bs', X_[y_==1].T[0], X_[y_==1].T[1],'ro')
subfig[3].axis([0, 10, 0, 10])
subfig[3].set_title('RENN')


plt.show()
