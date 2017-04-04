#coding:UTF-8
# 欧氏距离：
#Euclidean_distance
from math import sqrt

def Euclidean_distance(vector1,vector2):
    length = len(vector1)

    TSum = sum([pow((vector1[i] - vector2[i]),2) for i in range(len(vector1))])

    SSum = sqrt(TSum)

    return SSum