# -*-coding:utf-8-*-
from numpy import *
import numpy as np
from matplotlib import *
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

def loadDataSet(fileName):
	dataMat = []
	fr = open(fileName)
	for line in fr.readlines():
		curLine = line.strip().split('\t')
		fltLine = map(float,curLine)
		dataMat.append(fltLine)
	return dataMat
datMat=mat(loadDataSet('testSetRBF2.txt'))

plt.figure(figsize=(5, 5))

X=datMat

mydata= KMeans(n_clusters=4)
y_pred = mydata.fit_predict(X)

number=len(datMat)
for item in range(number):
	print item


plt.subplot(111)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("k-means")

plt.show()


# listmy=[1,2,3,4]
# listMean=mean(listmy)
# print listMean
