#coding:UTF-8
# 深度优先显示
def yezi(clust):
    if clust.left == None and clust.right == None :
        return [clust.id]
    # print "完成一层合并"
    return yezi(clust.left) + yezi(clust.right)