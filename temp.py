# -*-coding:utf-8-*-
import numpy as np 
import scipy.io as sio
import matplotlib.pyplot as plt

def test():
	data = []
	y_pred=[]
	x_pred_mat=[]
	y_pred_mat=[]
	x_pred_random=[]
	y_pred_random=[]
	data =sio.loadmat(u'five_cluster.mat')
	print data
	y_pred=data['y'][0]
	print y_pred
	x_pred=data['x']
	print x_pred
	n=len(y_pred)
	for i in range(n):
		dataMat=[x_pred[0][i],x_pred[1][i]]
		x_pred_mat.append(dataMat)
	for i in range(10):
		index=np.random.randint(0, n-1)
		x_pred_random.append(x_pred_mat[index])
		y_pred_random.append(y_pred[index])
	print x_pred_random,y_pred_random

test()

# b=a.split(' ')
# a=[0,1,2]
# b=a[0]
# print a