import numpy as np
import matplotlib.pyplot as plt

import env
import reward

class strategy(object):
	def __init__(self, machine, maxIter, b, d, delta):
		super(strategy, self).__init__()
		self.machine = machine
		self.maxIter = maxIter
		self.b = b
		self.d = d
		self.delta = delta
		self.numArm = self.machine.probList.size
		self.population = np.full(self.numArm, 1 / self.numArm, dtype = np.float64)
		self.prob = self.population / np.sum(self.population)
		self.probHistory = np.empty((self.maxIter, self.numArm), dtype = np.float64)
		self.time = 0
		return

	def getChoice(self):
		self.probHistory[self.time] = self.prob
		bound = np.cumsum(self.prob)
		sample = np.random.rand()
		choice = np.searchsorted(bound, sample)
		return choice

	def updateProb(self, choice, reward):
		factor = self.b * (reward - self.d * np.power(self.population[choice], self.delta))
		self.population[choice] = np.max([self.population[choice] + factor / (1 - factor) * np.sum(self.population), self.population[choice] * 0.1])
		self.prob = self.population / np.sum(self.population)
		self.time = self.time + 1
		return

	def play(self, iters):
		for i in range(iters):
			choice = self.getChoice()
			reward = self.machine.pull(choice)
			self.updateProb(choice, reward)
		return

	def plot(self, iters):
		plt.figure(2)
		self.machine.showReward()
		x = np.array(range(iters))
		color = ['b', 'g', 'r', 'c', 'c', 'm', 'y', 'k']
		for i in range(self.numArm):
			plt.plot(x, self.probHistory[:iters, i], color[i])
		return

if __name__ == '__main__':
	rewardList = [0.1, 0.3, 0.5, 0.9]
	probList = [0.3, 0.4, 0.1, 0.2]
	interval = 15000
	probClass = reward.cyclicalR
	maxIter = 60000
	M = env.machine(rewardList = rewardList, probList = probList, interval = interval, probClass = probClass, maxIter = maxIter)
	S = strategy(machine = M, maxIter = maxIter, b = 0.01, d = 0.05, delta = 0.15)

	S.play(60000)
	S.plot(60000)
	S.machine.history.plot()
	print(S.prob)
	plt.show()