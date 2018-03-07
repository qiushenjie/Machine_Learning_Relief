import numpy as np
import pandas as pd
from numpy import *

def dataGet(filename):
	df = pd.read_excel(filename)
	data = df.values[:, 1:-1]
	label = df.values[:, -1:]
	return data, label

'''
计算距离，包括：
1.首先求抽样点分别与猜中近邻、猜错近邻的欧式距离得到二者相对的近邻编号
2.再根据近邻编号求抽样点与其对应样本数据在属性j上diff距离，从而计算属性分量
'''
def calDistance(sample, Y):
	res = 0.0
	for i in range(len(sample)):
		sampleI = sample[i]
		YI = Y[i]
		if type(sample[i]).__name__=='float64':
			res += np.abs(sampleI-YI)
		else:
			res += 0 if sampleI == YI else 1
	return res

#寻找最近邻函数
def findNear(data, dataLabel, Xi, XiNum):
	m, n = np.shape(data)
	near_hit_dist = Inf
	near_miss_dist = Inf
	for i in range(m):
		if XiNum == i: continue
		dist = calDistance(Xi, data[i])
		if dataLabel[XiNum] == dataLabel[i]:
			if dist < near_hit_dist:
				near_hit = i
				near_hit_dist = dist
		else:
			if dist < near_miss_dist:
				near_miss = i
				near_miss_dist = dist
	return near_hit, near_miss

#relief算法主函数
def relief(data, dataLabel, MM):
	m, n = np.shape(data)
	weight = np.zeros((n))
	for i in range(MM):
		XiNum = random.randint(0, m-1)
		Xi = data[XiNum]
		near_hit_id, near_miss_id = findNear(data, dataLabel, Xi, XiNum)
		for j in range(n):
			weight[j] += - calDistance([Xi[j]], [data[near_hit_id][j]])/MM + calDistance([Xi[j]], [data[near_miss_id][j]])/MM
	print(weight)

if __name__=="__main__": 
	dataSet, label = dataGet('watermelon_4.3_one_hot.xlsx')
	#将连续型属性规范到[0, 1]区间
	dataSet[:, -2:-1] = (dataSet[:, -2:-1] - min(dataSet[:, -2:-1]))/(max(dataSet[:, -2:-1]) - min(dataSet[:, -2:-1]))
	dataSet[:, -1:] = (dataSet[:, -1:] - min(dataSet[:, -1:]))/(max(dataSet[:, -1:]) - min(dataSet[:, -1:]))
	relief(dataSet, label, 100)