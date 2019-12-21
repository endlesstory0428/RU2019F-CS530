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
		self.delta = delta + 1
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
		# factor = self.b * (reward - self.d * np.power(self.population[choice], self.delta))
		# self.population[choice] = np.max([self.population[choice] + factor / (1 - factor) * np.sum(self.population), self.population[choice] * 0.1])
		dec = self.b * self.d * np.power(self.population, self.delta)
		inc = self.b * reward
		self.population = np.max([self.population - dec, self.population * 0.1], axis = 0)
		self.population[choice] = self.population[choice] + inc / (1 - inc) * np.sum(self.population)
		self.prob = self.population / np.sum(self.population)
		self.time = self.time + 1
		return

	def play(self, iters):
		for i in range(iters):
			choice = self.getChoice()
			reward = self.machine.pull(choice)
			self.updateProb(choice, reward)
		return

	def plot(self, iters, history = None):
		if history is None:
			history = self.probHistory
		plt.figure(2)
		self.machine.showReward()
		x = np.array(range(iters))
		color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
		lineList = []
		labelList = []
		for i in range(self.numArm):
			line, = plt.plot(x, history[:iters, i], color[i])
			label = 'rate ' + str(i+1)
			lineList.append(line)
			labelList.append(label)
		plt.legend(handles = lineList, labels = labelList, loc = 2)
		plt.ylim(ymin = 0, ymax = 1)
		plt.yticks(np.arange(0, 1, 0.05))
		return

def getAverage(iters):
	rewardList = [0.1, 0.3, 0.5, 0.9]
	probList = [0.2, 0.3, 0.4, 0.1]
	interval = 20000
	probClass = reward.cyclicalR
	maxIter = 60000
	fullProbHistory = np.empty((iters, maxIter, len(probList)))
	fullCumReward = np.empty(iters)
	for i in range(iters):
		M = env.machine(rewardList = rewardList, probList = probList, interval = interval, probClass = probClass, maxIter = maxIter)
		S = strategy(machine = M, maxIter = maxIter, b = 0.05, d = 0.9, delta = 0.2)
		S.play(maxIter)
		fullProbHistory[i] = S.probHistory
		fullCumReward[i] = S.machine.history.cumReward

	history = np.average(fullProbHistory, axis = 0)
	cumReward = np.average(fullCumReward)
	wholeCumReward = np.sum(S.machine.history.wholeReward, axis = 1)
	randCumReward = np.sum(np.average(S.machine.history.wholeReward, axis = 0))
	rand3CumReward = np.sum(np.average(S.machine.history.wholeReward[:3], axis = 0))
	print('algorithm:')
	print(cumReward / S.machine.history.cumOptReward)
	print('stick on one rate:')
	print(wholeCumReward / S.machine.history.cumOptReward)
	print('random select:')
	print(randCumReward / S.machine.history.cumOptReward)
	print('random in first 3:')
	print(rand3CumReward / S.machine.history.cumOptReward)
	S.plot(maxIter, history)
	S.machine.history.plot(maxIter)
	plt.show()

if __name__ == '__main__':
	getAverage(10)