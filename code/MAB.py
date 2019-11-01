import numpy as np
import matplotlib.pyplot as plt

import env

class strategy(object):
	def __init__(self, machine, maxIter, b, d, delta):
		super(strategy, self).__init__()
		self.machine = machine
		self.maxIter = maxIter
		self.b = b
		self.d = d
		self.delta = delta
		self.numArm = self.machine.probList.size
		self.prob = np.full(self.numArm, 1 / self.numArm, dtype = np.float64)
		self.probHistory = np.empty((self.maxIter, self.numArm), dtype = np.float64)
		self.time = 0
		return

	def getChoice(self):
		self.probHistory[time] = self.prob
		bound = np.cumsum(self.prob)
		sample = np.random.rand()
		choice = np.searchsorted(bound, sample)
		return choice

	def updateProb(self, choice, reward):
		dProb = self.prob * self.b * reward
		dProb[choice] = dProb[choice] - np.sum(dProb)
		self.prob = self.prob - dProb
		self.time = self.time + 1
		return

	def play(self, iters):
		for i in range(iters):
			choice = self.getChoice
			reward = self.machine.pull(choice)
			self.updateProb(choice, reward)
		return

	def plot(self, iters):
		self.machine.showReward()
		return
