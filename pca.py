import matplotlib.pyplot as plt
import arff
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
import sys
import pylab as pl
from sklearn.cluster import KMeans
from reg_sci import regression
from mpl_toolkits.mplot3d import Axes3D
dataset = arff.load(open(sys.argv[1], 'rb'))
data = np.array(dataset['data']).astype(np.float64)

print(data[0,:])

# count = 0
# for i in range(0,1151):
# 	if(data[i,19]==1):
# 		count = count + 1
# a = np.zeros((count,20)).reshape(count,20)
# for i in range(0,count):
# 	if(data[i,19]==1):
# 		for b in range(0,20):
# 			a[i,b]=data[i,b]

# X = a[:,0:18].astype(np.float64)
# print(X.shape)
# y = a[:,19].astype(np.float64)
# print(y.shape)



X = data[:,0:18].astype(np.float64)
y = data[:,19].astype(np.float64)
target_names = ['Data Point']

# #KMeans
# pca = PCA(n_components=2).fit(X)
# pca_2d = pca.transform(X)
# pl.figure('Reference Plot')
# # pl.scatter(pca_2d[:, 0], pca_2d[:, 1], c=y)
# kmeans = KMeans(n_clusters=2, random_state=111)
# kmeans.fit(X)
# pl.figure('K-means with 2 clusters')
# pl.scatter(pca_2d[:, 0], pca_2d[:, 1], c=kmeans.labels_)
# pl.show()





#PCA

# pca = PCA(n_components=0)
# X_r = pca.fit(X).transform(X)


#Percentage of variance explained for each components
# print('explained variance ratio (first two components): %s'
#       % str(pca.explained_variance_ratio_))

# plt.figure()
# colors = ['navy']
# lw = 2

# for color, i, target_name in zip(colors, [0, 1, 2], target_names):
#     plt.scatter(X_r[y == i, 0], X_r[y == i,1], color=color, alpha=.8, lw=lw,
#                 label=target_names[i])
# plt.legend(loc='best', shadow=False, scatterpoints=1)
# plt.title('PCA of Diabetic Retinopathy Dataset')
# plt.xlabel("PC1");
# plt.ylabel("PC2");
# plt.grid();

# plt.show()
# print X_r.shape
regression(X_r[0:920,:],data[0:920,19],X_r[920:1151,:],data[920:1151,19])