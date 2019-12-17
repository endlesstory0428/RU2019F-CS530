import numpy as np
import matplotlib.pyplot as plt

import reward as rwd

class machineHistory(object):
	def __init__(self, rewardList, maxIter):
		super(machineHistory, self).__init__()
		self.rewardList = rewardList
		self.maxIter = maxIter
		self.choice = np.empty(self.maxIter, dtype = np.int8)
		self.reward = np.empty(self.maxIter, dtype = np.float64)
		self.optChoice = np.empty(self.maxIter, dtype = np.int8)
		self.optReward = np.empty(self.maxIter, dtype = np.float64)
		self.wholeReward = np.empty((self.rewardList.size, self.maxIter), dtype = np.float64)
		self.cumReward = 0
		self.cumOptReward = 0
		return

	def getOpt(self, probList):
		tempReward = self.rewardList * probList
		optIdx = np.argmax(tempReward)
		return optIdx, tempReward

	def uptade(self, time, choice, result, probList):
		tempReward = result
		optIdx, tempWholeReward = self.getOpt(probList)
		self.choice[time] = choice
		self.reward[time] = tempReward
		self.optChoice[time] = optIdx
		self.optReward[time] = tempWholeReward[optIdx]
		self.wholeReward[:, time] = tempWholeReward
		self.cumReward = self.cumReward + tempReward
		self.cumOptReward = self.cumOptReward + tempWholeReward[optIdx]
		return

	def plot(self, time):
		plt.figure(1)
		color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
		x = np.array(range(time), dtype = np.int32)
		for i in range(self.rewardList.size):
			mask = np.zeros(time, dtype = np.bool)
			index = np.where(self.optChoice[:time] == i)
			mask[index] = True
			plt.scatter(x[mask], self.wholeReward[i][mask], c = color[i], marker = 'o', s = 10)
			plt.scatter(x[~mask], self.wholeReward[i][~mask], c = color[i], marker = 'o', s = 0.2)
		return
		

class machine(object):
	def __init__(self, rewardList, probList, interval, probClass, maxIter):
		super(machine, self).__init__()
		self.probList = np.array(probList, dtype = np.float64)
		self.rewardList = np.array(rewardList[::-1], dtype = np.float64)
		self.interval = interval
		self.probClass = probClass
		self.maxIter = maxIter
		self.prob = self.probClass(self.probList, self.interval)
		self.time = 0
		self.history = machineHistory(self.rewardList, self.maxIter)
		return

	def pull(self, choice):
		tempProb = self.prob.getMu(self.time)
		result = (np.random.rand() < tempProb[choice]) * self.rewardList[choice]
		self.history.uptade(self.time, choice, result, tempProb)
		self.time = self.time + 1
		return result

	def showReward(self):
		tempProb = self.prob.getMu(self.time)
		print('prob:')
		print(tempProb)
		print('reward:')
		print(self.rewardList)
		print('exp:')
		print(self.rewardList * tempProb)
		return