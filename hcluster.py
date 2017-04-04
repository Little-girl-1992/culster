#coding:UTF-8
#Hierarchical clustering 层次聚类

import numpy as np
from matplotlib import *
import scipy.io as sio
import random as random
import matplotlib.pyplot as plt

#深度优先显示：
def yezi(clust):
	if clust.left == None and clust.right == None :
		return [clust.id]				#如果clust没有叶子结点，则输出clust的id。
	return yezi(clust.left) + yezi(clust.right)		#一旦有左结点或者右结点就递归遍历。


#欧氏距离：
#Euclidean_distance
from math import sqrt
def Euclidean_distance(vector1,vector2):
	length = len(vector1)
	TSum = sum([pow((vector1[i] - vector2[i]),2) for i in range(len(vector1))])
	SSum = sqrt(TSum)
	return SSum

'''
#从文件中读取数据集
def loadDataSet(fileName):
	dataMat = []
	fr = open(fileName)
	for line in fr.readlines():
		curLine = line.strip().split('\t')
		fltLine = map(float, curLine)			#map all elements to float()
		dataMat.append(fltLine)
	return dataMat
'''

class bicluster:				#定义聚类的数据结构，left，right相当于其子聚类，想象下树的结构。
	def __init__(self, vec, left=None,right=None,distance=0.0,id=None):
		self.left = left
		self.right = right  #每次聚类都是一对数据，left保存其中一个数据，right保存另一个
		self.vec = vec      #保存两个数据聚类后形成新的中心,vec是一个向量，表示此聚类的聚类中心
		self.id = id     
		self.distance = distance

#在matplotlib中显示
def pucture(X,l):						#X是集合类型，L为列表类型
	plt.figure(figsize=(5, 5))
	y_pred=l
	plt.subplot(111)
	plt.scatter(X[:, 0], X[:, 1], c=y_pred)				#显示
	plt.title("clustering")
	plt.show()

#层次聚类函数
def hcluster(blogwords,numOfCluster) :					#参数blogwords表示输入的数据，numOfCluster=k表示聚类个数
	#biclusters存放所有聚类
	biclusters = [ bicluster(vec = blogwords[i], id = i ) for i in range(len(blogwords)) ]
	distances = {}					#字典
	flag = None;
	currentclusted = -1
	while(len(biclusters) > numOfCluster) : #如果当前聚类总数>要求聚类个数numOfCluster时，继续合并聚类
		min_val = 999999999999 #Python的无穷大应该是inf
		biclusters_len = len(biclusters)
		for i in range(len(biclusters)-1) :				#i表示聚类中的第i个聚类
			for j in range(i + 1, biclusters_len) :			#比较第i个聚类和剩余聚类之间的距离
				if distances.get((biclusters[i].id,biclusters[j].id)) == None:
					#两个聚类之间的距离用聚类中心的距离表示
					distances[(biclusters[i].id,biclusters[j].id)] = Euclidean_distance(biclusters[i].vec,biclusters[j].vec)
				d = distances[(biclusters[i].id,biclusters[j].id)] 
				if d < min_val :
					min_val = d				#更新最小距离
					flag = (i,j)			#应该合并的两个聚类是聚类i和j
		bic1,bic2 = flag					#解包bic1 = i , bic2 = j
		#求bic1,bic2合并之后的聚类的中心。
		newvec = [(biclusters[bic1].vec[i] + biclusters[bic2].vec[i])/2 for i in range(len(biclusters[bic1].vec))]
		newbic = bicluster(newvec, left=biclusters[bic1], right=biclusters[bic2], distance=min_val, id = currentclusted) #二合一

		currentclusted -= 1
		del biclusters[bic2] #删除聚成一起的两个数据，由于这两个数据要聚成一起
		del biclusters[bic1]
		biclusters.append(newbic)				#将新合并的聚类追加到biclusters之后
		clusters = [yezi(biclusters[i]) for i in range(len(biclusters))] #深度优先搜索叶子节点，用于输出显示
	return clusters # biclusters				#返回一个列表，列表的一个项表示一个聚类。


'''
def show(dataSet, k, centroids, clusterAssment):
	numSamples, dim = dataSet.shape
	#plt.figure(figsize = (10,10), dpi=100)

	p1 = plt.subplot(121)
	p2 = plt.subplot(122)

	p1.set_title("Before",fontsize=15)
	#p2.set_title("k-means    K=4",fontsize=15)
	p2.set_title("bi-k-means   K=5",fontsize=15)
	#打印原始点
	for i in range(shape(dataSet)[0]):						
		p1.plot(dataSet[i, 0], dataSet[i, 1], 'Hr', markersize = 10)
	mark = ['hr', 'hb', 'hm', 'hg', 'hc', '+r', 'sr', 'dr', '<r', 'pr']
	#打印聚类之后的点
	for i in xrange(numSamples):
		markIndex = int(clusterAssment[i, 0])  
		p2.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex],markersize = 10)
	#打印聚类中心
	#mark = ['*r', '*b', '*m', '*g','*c' '^b', '+b', 'sb', 'db', '<b', 'pb']
	for i in range(k):
		p2.plot(centroids[i, 0], centroids[i, 1],'*k', markersize = 15)
	plt.show()
	'''


#执行层次聚类
def test2():
	data = []
	y_pred=[]
	y_pred_random=[]
	dataMatTal =sio.loadmat(u'Twomoons.mat') 
	dataMat=dataMatTal['Twomoons']
	# dataMatTal =sio.loadmat(u'E:\ThreeCircles.mat')			#使用matlab的数据
	# dataMat=dataMatTal['ThreeCircles']					#？？？？

	# dataMat = mat(loadDataSet(r'E:\testSet4.txt'))

	data=[pointData[1:3] for pointData in dataMat]
	n=len(data)
	for i in range(500):
		index=random.randint(0, n-1)
		y_pred_random.append(data[index])				#将dataMat中的数据最终转化保存y_pred_random列表中
	print y_pred_random
	print "完成数据转换"
	l=hcluster(y_pred_random, 3)				#调用hcluster完成聚类，参数为处理过的数据以及聚类个数k，层次聚类函数返回值是一个二维列表。
	print l
	print "完成聚类过程"
	dist=[0]*500			#定义列表dist，对聚类的返回结果L进行处理
	for i in range(len(l)):
		for j in range(len(l[i])):
			dist[l[i][j]]=i						#
	y_pred=dist
	print y_pred
	print "开始画图"
	y_pred_random=np.mat(y_pred_random)				#y_pred_random由列表转换为集合类型
	pucture(y_pred_random, y_pred)
	print "结束"
test2()