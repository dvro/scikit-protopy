import numpy as np
import matplotlib.pyplot as plt

from protopy.selection.enn import ENN

mu1 = [4, 5]
si1 = [[0.75, 0.25], [0.25, 0.75]]

mu2 = [6, 5]
si2 = [[0.25, 0.75], [0.75, 0.25]]

samples = 200

X1 = np.random.multivariate_normal(
    np.asarray(mu1), np.asarray(si1), samples)
X2 = np.random.multivariate_normal(
    np.asarray(mu2), np.asarray(si2), samples)

X = np.vstack((X1, X2))

y = np.asarray([0] * samples + [1] * samples)


plt.plot(X[y==0].T[0], X[y==0].T[1], 'bs', X[y==1].T[0], X[y==1].T[1],'ro')
plt.axis([0, 10, 0, 10])
plt.title('Original Dataset')
plt.show()
plt.clf()

editednn = ENN()
X_, y_ = editednn.reduce_data(X, y)

plt.plot(X_[y_==0].T[0], X_[y_==0].T[1], 'bs', X_[y_==1].T[0], X_[y_==1].T[1],'ro')
plt.axis([0, 10, 0, 10])
plt.title('ENN')
plt.show()
plt.clf()


