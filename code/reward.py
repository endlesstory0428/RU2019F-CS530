import numpy as np
import matplotlib.pyplot as plt

class constR(object):
	def __init__(self, probList, interval = None):
		super(constR, self).__init__()
		self.probList = np.array(probList, dtype = np.float64)
		self.interval = interval
		self._constructProbFunc()
		return

	def _constructProbFunc(self):
		def getProbFunc(probList, interval = None, bias = None):
			normProbList = probList / np.sum(probList)
			def probFunc(time):
				return normProbList
			return probFunc

		self.probFunc = getProbFunc(self.probList)
		return self.probFunc

	def getMu(self, time):
		mu = np.cumsum(self.probFunc(time))
		return mu


class cyclicalR(object):
	def __init__(self, probList, interval = 15000):
		super(cyclicalR, self).__init__()
		self.probList = np.array(probList, dtype = np.float64)
		self.interval = interval
		self._constructProbFunc()
		return

	def _constructProbFunc(self):
		def getProbFunc(probList, interval, bias = 0):
			normProbList = probList / np.sum(probList)
			probSize = probList.size
			extProbList = np.empty(probList.size * 2 - 1, dtype = probList.dtype)
			extProbList[:probSize] = normProbList
			extProbList[probSize:] = normProbList[:probSize - 1]
			def probFunc(time):
				index = (time // interval + bias) % probSize
				return extProbList[index: index + probSize]
			return probFunc
		self.probFunc = getProbFunc(self.probList, self.interval)
		return

	def getMu(self, time):
		mu = np.cumsum(self.probFunc(time))
		return mu


class cosinR(object):
	def __init__(self, probList, interval = 15000):
		super(cosinR, self).__init__()
		self.probList = np.array(probList, dtype = np.float64)
		self.interval = np.pi / interval * 2
		self._constructProbFunc()
		return

	def _constructProbFunc(self):
		def getProbFunc(probList, interval, bias = 0):
			probSize = probList.size
			phase = ((np.array(range(probSize), dtype = np.float64) + bias) % probSize) / probSize * 2 * np.pi
			def probFunc(time):
				tempProb = probList * (2 + np.cos(time * interval + phase))
				return tempProb / np.sum(tempProb)
			return probFunc
		self.probFunc = getProbFunc(self.probList, self.interval)
		return

	def getMu(self, time):
		mu = np.cumsum(self.probFunc(time))
		return mu

if __name__ == '__main__':
	machine = cosinR([3, 4, 5, 6])
	reward = np.empty((15000, 4), dtype = np.float64)
	for i in range(15000):
		reward[i] = machine.getMu(i)
	x = np.array(range(15000))
	plt.plot(x, reward[:, 0], 'r')
	plt.plot(x, reward[:, 1] - reward[:, 0], 'g')
	plt.plot(x, reward[:, 2] - reward[:, 1], 'b')
	plt.plot(x, reward[:, 3] - reward[:, 2], 'k')
	plt.show()