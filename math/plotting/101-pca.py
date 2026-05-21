#!/usr/bin/env python3
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

lib = np.load("pca.npz")
data = lib["data"]
labels = lib["labels"]

data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)

# 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# scatter plot in 3D using PCA results
sc = ax.scatter(
    pca_data[:, 0],
    pca_data[:, 1],
    pca_data[:, 2],
    c=labels,
    cmap='plasma'
)

# axis labels
ax.set_xlabel('U1')
ax.set_ylabel('U2')
ax.set_zlabel('U3')

# title
ax.set_title('PCA of Iris Dataset')

# colorbar
plt.colorbar(sc)

# show plot
plt.show()