import numpy as np

class constR(object):
	def __init__(self, rewardList, interval = None, sigma = 1):
		super(constR, self).__init__()
		self.rewardList = rewardList
		self.interval = interval
		self.sigma = sigma
		self.rewardFuncList = self._constructRewardFuncList()
		return

	def _constructRewardFuncList(self):
		def constructRewardFunc(config):
			def rewardFunc(time):
				return config
			return rewardFunc

		self.rewardFuncList = []
		for reward in self.rewardList:
			self.rewardFuncList.append(constructRewardFunc(reward))
		return self.rewardFuncList

	def getReward(self, arm, time):
		mu = self.rewardFuncList[arm](time)
		return np.random.normal(mu, np.sqrt(self.sigma))


class cyclicalR(object):
	def __init__(self, rewardList, interval = 15000, sigma = 1):
		super(cyclicalR, self).__init__()
		self.rewardList = rewardList
		self.interval = interval
		self.sigma = sigma
		self.rewardFuncList = self._constructRewardFuncList()
		return

	def _constructRewardFuncList(self):
		def constructRewardFunc(config):
			def rewardFunc(time):
				rewardList, interval, bias = config
				index = (time // interval + bias) % len(rewardList)
				return rewardList[index]
			return rewardFunc

		self.rewardFuncList = []
		for i in range(len(self.rewardList)):
			conifg = (self.rewardList, self.interval, i)
			self.rewardFuncList.append(constructRewardFunc(conifg))
		return self.rewardFuncList

	def getReward(self, arm, time):
		mu = self.rewardFuncList[arm](time)
		return np.random.normal(mu, np.sqrt(self.sigma))


class cosineR(object):
	def __init__(self, rewardList, interval = 15000, sigma = 1):
		super(cosineR, self).__init__()
		self.rewardList = rewardList
		self.interval = np.pi / interval * 2
		self.sigma = sigma
		self.rewardFuncList = self._constructRewardFuncList()
		return

	def _constructRewardFuncList(self):
		def constructRewardFunc(config):
			def rewardFunc(time):
				reward, interval, bias = config
				return reward * (2 + np.cos(time * interval + bias * (np.pi / 2)))
			return rewardFunc

		self.rewardFuncList = []
		for i, reward in enumerate(self.rewardList):
			conifg = (reward, self.interval, i)
			self.rewardFuncList.append(constructRewardFunc(conifg))
		return self.rewardFuncList

	def getReward(self, arm, time):
		mu = self.rewardFuncList[arm](time)
		return np.random.normal(mu, np.sqrt(self.sigma))

if __name__ == '__main__':
	machine = cosineR([3, 4, 5, 6])
	reward = []
	for i in range(1000):
		reward.append(machine.getReward(0, 3750))
	print(np.mean(reward))