#coding:UTF-8
"""
=============================================================================
Various Agglomerative Clustering on a 2D embedding of digits
=============================================================================

An illustration of various linkage option for agglomerative clustering on
a 2D embedding of the digits dataset.

The goal of this example is to show intuitively how the metrics behave, and
not to find good clusters for the digits. This is why the example works on a
2D embedding.

What this example shows us is the behavior "rich getting richer" of
agglomerative clustering that tends to create uneven cluster sizes.
This behavior is especially pronounced for the average linkage strategy,
that ends up with a couple of singleton clusters.
"""

# Authors: Gael Varoquaux
# License: BSD 3 clause (C) INRIA 2014

print(__doc__)
from time import time
import scipy.io as sio
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt

from sklearn import manifold, datasets

# digits = datasets.load_digits(n_class=10)
# X = digits.data
# y = digits.target
# n_samples, n_features = X.shape

np.random.seed(0)

def nudge_images(X, y):
    # Having a larger dataset shows more clearly the behavior of the
    # methods, but we multiply the size of the dataset only by 2, as the
    # cost of the hierarchical clustering methods are strongly
    # super-linear in n_samples
    shift = lambda x: ndimage.shift(x.reshape((8, 8)),
                                  .3 * np.random.normal(size=2),
                                  mode='constant',
                                  ).ravel()
    X = np.concatenate([X, np.apply_along_axis(shift, 1, X)])
    Y = np.concatenate([y, y], axis=0)
    return X, Y


# X, y = nudge_images(X, y)

def test1():

    data = []
    data_pred=[]
    data_y_pred=[]
    y_pred_random=[]
    # dataMatTal =sio.loadmat(u'Twomoons.mat') 
    # dataMat=dataMatTal['Twomoons']
    # dataMatTal =sio.loadmat(u'ThreeCircles.mat') 
    # dataMat=dataMatTal['ThreeCircles']
    dataMatTal =sio.loadmat(u'spiral.mat') 
    dataMat=dataMatTal['spiral']
    n=len(dataMat)
    print ("共有%d个点"%n)
    for i in range(3000):
        index=np.random.randint(0, n-1)
        y_pred_random.append(dataMat[index])
    # print y_pred_random
    for pointData in y_pred_random:
        data=pointData[1:3]
        data_y=pointData[0]-1
        data_pred.append(data)
        data_y_pred.append(data_y)
    data_mat=np.mat(data_pred)
    data_y_mat=map(int,data_y_pred)
    print "完成数据转换"
    return data_mat,data_y_mat

def test():
    data = []
    y_pred=[]
    x_pred_mat=[]
    x_pred_random=[]
    y_pred_random=[]
    data =sio.loadmat(u'five_cluster.mat')
    # print data
    # y_pred=data['y'][0]-1
    y_pred=data['y'][0]
    # print y_pred
    x_pred=data['x']
    # print x_pred
    n=len(y_pred)
    print n
    for i in range(n):
        dataMat=[x_pred[0][i],x_pred[1][i]]
        x_pred_mat.append(dataMat)
    for i in range(2000):
        index=np.random.randint(0, n-1)
        x_pred_random.append(x_pred_mat[index])
        y_pred_random.append(y_pred[index])
    return x_pred_random,y_pred_random


X,y=test1()
# print y
#----------------------------------------------------------------------
# Visualize the clustering
def plot_clustering(X_red,labels, title=None):
    x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
    X_red = (X_red - x_min) / (x_max - x_min)

    plt.figure(figsize=(5, 5))
    for i in range(X_red.shape[0]):
        plt.text(X_red[i, 0], X_red[i, 1],str(labels[i]),
                 color=plt.cm.spectral(labels[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, size=17)
    plt.axis('off')
    plt.tight_layout()

#----------------------------------------------------------------------
# 2D embedding of the digits dataset
print("Computing embedding")
X_red = X
print("Done.")

from sklearn.cluster import AgglomerativeClustering,MiniBatchKMeans,DBSCAN
from sklearn.neighbors import kneighbors_graph
# connectivity matrix for structured Ward
connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
# make connectivity symmetric
connectivity = 0.5 * (connectivity + connectivity.T)

clustering =DBSCAN(eps=.2)
clustering.fit(X_red)
plot_clustering(X_red,clustering.labels_,title="DBSCAN")

# clustering = MiniBatchKMeans(n_clusters=3)
# clustering.fit(X_red)
# plot_clustering(X_red,clustering.labels_,title="kMeans")


# for linkage in ('average', 'ward','complete'):
#     if linkage != 'ward':
#         clustering = AgglomerativeClustering(linkage=linkage,affinity="cityblock",connectivity=connectivity, n_clusters=3)
#     else:
#         clustering = AgglomerativeClustering(n_clusters=3, linkage='ward',connectivity=connectivity)
#     t0 = time()
#     clustering.fit(X_red)
#     print("%s : %.2fs" % (linkage, time() - t0))

#     plot_clustering(X_red,clustering.labels_, "%s linkage" % linkage)
#     yy=np.mat(clustering.labels_)
#     smstr = np.nonzero(y-yy);
#     num_error=np.shape(smstr[0])[1]
#     print num_error
#     error_rate=num_error*1.0/len(y)
#     print "%s linkage of error_rate" % linkage,error_rate
# plot_clustering(X, y,title="initial data")

plt.show()
