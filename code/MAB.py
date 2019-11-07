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
		self.delta = 1 + delta
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
		# dProb = self.prob * self.b * reward
		# dProb[choice] = dProb[choice] - np.sum(dProb)
		dPopulation = np.zeros_like(self.prob, dtype = np.float64)
		factor = self.b * (reward -self.d * np.power(self.population[choice], self.delta))
		dPopulation[choice] = dPopulation[choice] + factor / (1 - factor) 
		self.population = self.population + dPopulation
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
		self.machine.showReward()
		x = np.array(range(iters))
		color = ['b', 'g', 'r', 'c', 'c', 'm', 'y', 'k']
		for i in range(self.numArm):
			plt.plot(x, self.probHistory[:iters, i], color[i])
		plt.show()
		return

if __name__ == '__main__':
	rewardList = [0.1, 0.3, 0.5, 0.9]
	probList = [0.4, 0.3, 0.2, 0.1]
	interval = 15000
	probClass = reward.constR
	maxIter = 15000
	M = env.machine(rewardList = rewardList, probList = probList, interval = interval, probClass = probClass, maxIter = maxIter)
	S = strategy(machine = M, maxIter = maxIter, b = 0.05, d = 0.5, delta = 0.5)

	S.play(15000)
	S.plot(15000)
	print(S.prob)