#coding:UTF-8
#Hierarchical clustering 层次聚类
import numpy as np
from matplotlib import *
import scipy.io as sio
import random as random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from Euclidean_distance import Euclidean_distance
from yezi import yezi
class bicluster:
    def __init__(self, vec, left=None,right=None,distance=0.0,id=None):
        self.left = left
        self.right = right  #每次聚类都是一对数据，left保存其中一个数据，right保存另一个
        self.vec = vec      #保存两个数据聚类后形成新的中心
        self.id = id     
        self.distance = distance
        
def hcluster(blogwords,n) :
    biclusters = [ bicluster(vec = blogwords[i], id = i ) for i in range(len(blogwords)) ]
    distances = {}
    flag = None;
    currentclusted = -1
    while(len(biclusters) > n) : #假设聚成n个类
        min_val = 999999999999; #Python的无穷大应该是inf
        biclusters_len = len(biclusters)
        #求相似度矩阵，求距离最小的两点
        for i in range(biclusters_len-1) :
            for j in range(i + 1, biclusters_len) :
                if distances.get((biclusters[i].id,biclusters[j].id)) == None:
                    distances[(biclusters[i].id,biclusters[j].id)] = Euclidean_distance(biclusters[i].vec,biclusters[j].vec)
                d = distances[(biclusters[i].id,biclusters[j].id)] 
                if d < min_val :
                    min_val = d
                    flag = (i,j)
        bic1,bic2 = flag #解包bic1 = i , bic2 = j
        newvec = [(biclusters[bic1].vec[i] + biclusters[bic2].vec[i])/2 for i in range(len(biclusters[bic1].vec))] #形成新的类中心，平均
        newbic = bicluster(newvec, left=biclusters[bic1], right=biclusters[bic2], distance=min_val, id = currentclusted) #二合一
        currentclusted -= 1
        del biclusters[bic2] #删除聚成一起的两个数据，由于这两个数据要聚成一起
        del biclusters[bic1]
        biclusters.append(newbic)#补回新聚类中心
        clusters = [yezi(biclusters[i]) for i in range(len(biclusters))] #深度优先搜索叶子节点，用于输出显示
    return clusters #biclusters,


def h_s_cluster(blogwords,n) :
    biclusters = [ bicluster(vec = blogwords[i], id = i ) for i in range(len(blogwords)) ]
    distances = {}
    flag = None;
    currentclusted = -1
    while(len(biclusters) > n) : #假设聚成n个类
        min_val = 999999999999; #Python的无穷大应该是inf
        biclusters_len = len(biclusters)
        #求相似度矩阵，求距离最小的两点
        for i in range(biclusters_len-1) :
            for j in range(i + 1, biclusters_len) :
                if distances.get((biclusters[i].id,biclusters[j].id)) == None:
                    distances[(biclusters[i].id,biclusters[j].id)] = Euclidean_distance(biclusters[i].vec,biclusters[j].vec)
                d = distances[(biclusters[i].id,biclusters[j].id)] 
                if d < min_val :
                    min_val = d
                    flag = (i,j)
        bic1,bic2 = flag #解包bic1 = i , bic2 = j
        newvec = [(biclusters[bic1].vec[i] + biclusters[bic2].vec[i])/2 for i in range(len(biclusters[bic1].vec))] #形成新的类中心，平均
        newbic = bicluster(newvec, left=biclusters[bic1], right=biclusters[bic2], distance=min_val, id = currentclusted) #二合一
        currentclusted -= 1
        del biclusters[bic2] #删除聚成一起的两个数据，由于这两个数据要聚成一起
        del biclusters[bic1]
        biclusters.append(newbic)#补回新聚类中心
        clusters = [yezi(biclusters[i]) for i in range(len(biclusters))] #深度优先搜索叶子节点，用于输出显示
    return clusters #biclusters,

def pucture(X,l):
    plt.figure(figsize=(5, 5))
    y_pred=l
    plt.subplot(111)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.title("clustering")

    plt.show()

def test3():
    data = []
    y_pred=[]
    y_pred_random=[]
    # dataMatTal =sio.loadmat(u'Twomoons.mat') 
    # dataMat=dataMatTal['Twomoons']
    dataMatTal =sio.loadmat(u'ThreeCircles.mat') 
    dataMat=dataMatTal['ThreeCircles']
    data=[pointData[1:3] for pointData in dataMat]
    n=len(data)
    for i in range(500):
        index=random.randint(0, n-1)
        y_pred_random.append(data[index])
    print y_pred_random
    print "完成数据转换"
    mydata= KMeans(n_clusters=3)
    l = mydata.fit_predict(y_pred_random)
    print l
    print "完成聚类过程"
    print "开始画图"
    y_pred_random=np.mat(y_pred_random)
    pucture(y_pred_random, l)
    print "结束"

def test2():
    data = []
    y_pred=[]
    y_pred_random=[]
    # dataMatTal =sio.loadmat(u'Twomoons.mat') 
    # dataMat=dataMatTal['Twomoons']
    dataMatTal =sio.loadmat(u'ThreeCircles.mat') 
    dataMat=dataMatTal['ThreeCircles']
    data=[pointData[1:3] for pointData in dataMat]
    n=len(data)
    for i in range(500):
        index=random.randint(0, n-1)
        y_pred_random.append(data[index])
    print y_pred_random
    print "完成数据转换"
    l=hcluster(y_pred_random, 3)
    print l
    print "完成聚类过程"
    dist=[0]*500
    for i in range(len(l)):
        for j in range(len(l[i])):
            dist[l[i][j]]=i
    y_pred=dist
    print y_pred
    print "开始画图"
    y_pred_random=np.mat(y_pred_random)
    pucture(y_pred_random, y_pred)
    print "结束"


def test1():
    data = []
    y_pred=[]
    f = open('testSetRBF2.txt','rb')
    for i in f.readlines():
        i=i.strip().split("\t")
        data.append(i)
    f.close()
    data=np.array(data).astype(float)
    n=len(data)
    k,l=hcluster(data, 4)
    dist=[0]*n
    for i in range(len(l)):
        for j in range(len(l[i])):
            dist[l[i][j]]=i
    y_pred=dist
    pucture(data, y_pred)

# test3()
test2()