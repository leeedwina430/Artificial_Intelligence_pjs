from mappings import *
from time import time
from itertools import product


class PriorSearch():
	""" Search before min max """

	def __init__(self, board):
		self.board = board

	def VcfStart(self):
		if self.board.step < 6:
			return -1
		for _type in [VCF6, VCF5, VCF4, VCF3, VCF2, VCF1]:
			if self.board.Cross[self.board.my][_type]:
				return list(self.board.Cross[self.board.my][_type])[0]
		return -1

	def Vcf1(self):
		startPoints = self.VcfStart()  # 找一个>=眠三的开始点
		if startPoints != -1:
			result = self.Vcf2(self.board.my, 0, startPoints)
			return result

		return 0

	# 二轮VCF，找到符合点开始第三轮VCF；赢的点记录winPoint；如果都没有/最大深度，返回0
	def Vcf2(self, searcher, depth, startPoint):
		self.board.best = (4, 4)

		if self.board.Cross[self.board.my][VCF6]:
			self.board.best = list(self.board.Cross[self.board.my][VCF6])[0]
			return 1
		if len(self.board.Cross[self.board.opp][VCF6]) >= 2:
			return -2

		# offend
		if self.board.Cross[self.board.opp][VCF6]:  # len() == 1
			oppPoint = list(self.board.Cross[self.board.opp][VCF6])[0]
			self.board.move(oppPoint[0], oppPoint[1])
			q = -self.Vcf3(searcher, depth + 1, startPoint)
			self.board.delmove()
			if q < 0:
				q -= 1
			elif q > 0:
				self.board.best = oppPoint
				q += 1
			return q

		# 己方活四，三步胜利
		if self.board.Cross[self.board.my][VCF5]:
			self.board.best = list(self.board.Cross[self.board.my][VCF5])[0]
			return 3

		# 己方冲四活三，尝试
		if self.board.my == searcher and self.board.Cross[self.board.my][VCF4]:
			# 对方没有能成四的点，五步胜
			if not (self.board.Cross[self.board.opp][VCF5] | self.board.Cross[self.board.opp][VCF4] |
					self.board.Cross[self.board.opp][VCF3] | self.board.Cross[self.board.opp][VCF2] |
					self.board.Cross[self.board.opp][VCF1]):
				self.board.best = list(self.board.Cross[self.board.my][VCF4])[0]
				return 5
			# 如果对方有能成四的点,对于己方能形成冲四+活三棋形的点一一尝试
			for point in self.board.Cross[self.board.my][VCF4]:
				self.board.move(point[0], point[1])
				q = -self.Vcf3(searcher, depth + 1, point)
				self.board.delmove()
				if q > 0:
					self.board.best = point
					return q + 1

		# 如果有冲四：则只在起始点（活三点）的八邻域寻找冲四点
		if self.board.my == searcher and depth < MAXFDEPTH and (
				self.board.Cross[self.board.my][VCF3] | self.board.Cross[self.board.my][VCF2]):
			for point in product(range(startPoint[0] - 4, startPoint[0] + 5),
								 range(startPoint[1] - 4, startPoint[1] + 5)):
				if point in (self.board.Cross[self.board.my][VCF3] | self.board.Cross[self.board.my][VCF2]):
					self.board.move(point[0], point[1])
					q = -self.Vcf3(searcher, depth + 1, point)
					self.board.delmove()
					if q > 0:
						self.board.best = point
						return q + 1

		# 己方双活三，对方没有任何活四冲四，五步胜
		if self.board.my == searcher and self.board.Cross[self.board.my][VCT4]:
			if not (self.board.Cross[self.board.opp][VCF5] | self.board.Cross[self.board.opp][VCF4] |
					self.board.Cross[self.board.opp][VCF3] | self.board.Cross[self.board.opp][VCF2] |
					self.board.Cross[self.board.opp][VCF1]):
				self.board.best = list(self.board.Cross[self.board.my][VCT4])[0]
				return 5

		return 0

	# 三轮VCF
	def Vcf3(self, searcher, depth, startpoint):
		# 己方有即将连五，1步赢
		if self.board.Cross[self.board.my][VCF6]:
			return 1

		# 对方有双连五，2步输
		if len(self.board.Cross[self.board.opp][VCF6]) >= 2:
			return -2

		# offend
		if self.board.Cross[self.board.opp][VCF6]:
			oppPoint = list(self.board.Cross[self.board.opp][VCF6])[0]
			self.board.move(oppPoint[0], oppPoint[1])
			q = -self.Vcf3(searcher, depth + 1, startpoint)
			self.board.delmove()
			q += -1 if q < 0 else 1
			return q

		# 己方活四，三步胜利
		if self.board.Cross[self.board.my][VCF5]:
			return 3

		# 己方有冲四活三
		if self.board.my == searcher and self.board.Cross[self.board.my][VCF4]:
			# 对方没有活四冲四，五步胜利；其他情况不予考虑（返回0）
			if not (self.board.Cross[self.board.opp][VCF5] | self.board.Cross[self.board.opp][VCF4] |
					self.board.Cross[self.board.opp][VCF3] | self.board.Cross[self.board.opp][VCF2] |
					self.board.Cross[self.board.opp][VCF1]):
				return 5

			for point in self.board.Cross[self.board.my][VCF4]:
				self.board.move(point[0], point[1])
				q = -self.Vcf3(searcher, depth + 1, point)
				self.board.delmove()
				if q > 0:
					return q + 1

		# 己方有冲四或其他棋型
		if self.board.my == searcher and depth < MAXFDEPTH and (
				self.board.Cross[self.board.my][VCF3] | self.board.Cross[self.board.my][VCF2]):
			for point in product(range(startpoint[0] - 4, startpoint[0] + 5),
								 range(startpoint[1] - 4, startpoint[1] + 5)):
				if point in (self.board.Cross[self.board.my][VCF3] | self.board.Cross[self.board.my][VCF2]):
					self.board.move(point[0], point[1])
					q = -self.Vcf3(searcher, depth + 1, point)
					self.board.delmove()
					if q > 0:
						return q + 1

		# 己方双活三，对方没有任何活四冲四，五步胜
		if self.board.my == searcher and self.board.Cross[self.board.my][VCT4]:
			if not (self.board.Cross[self.board.opp][VCF5] | self.board.Cross[self.board.opp][VCF4] |
					self.board.Cross[self.board.opp][VCF3] | self.board.Cross[self.board.opp][VCF2] |
					self.board.Cross[self.board.opp][VCF1]):
				return 5

		return 0

	################# VCF Done ####################
	################# VCT Start ###################

	def VctStart(self):
		if self.board.step < 4:
			return -1

		for _type in [VCT4, VCT3, VCT2, VCT1]:
			if self.board.Cross[self.board.my][_type]:
				return list(self.board.Cross[self.board.my][_type])[0]

		return -1

	# 一轮VCT
	def Vct1(self):
		VCT_startTime = time()
		startPoint = self.VctStart()

		result = 0
		if startPoint != -1:
			for depth in range(10, MAXTDEPTH + 2, 2):
				result = self.Vct2(self.board.my, 0, depth, startPoint)
				if result > 0 or 4000 * (time() - VCT_startTime) >= MAXTTIME:
					break

		return result

	# 二轮VCT
	def Vct2(self, searcher, depth, maxDepth, startPoint):
		self.board.best = (4, 4)
		if self.board.Cross[self.board.my][VCF6]:
			self.board.best = list(self.board.Cross[self.board.my][VCF6])[0]
			return 1
		if len(self.board.Cross[self.board.opp][VCF6]) >= 2:
			return -2

		# offend
		if self.board.Cross[self.board.opp][VCF6]:
			oppPoint = list(self.board.Cross[self.board.opp][VCF6])[0]
			self.board.move(oppPoint[0], oppPoint[1])
			q = -self.Vct3(searcher, depth + 1, maxDepth, startPoint)
			self.board.delmove()
			if q < 0:
				q -= 1
			elif q > 0:
				self.board.best = oppPoint
				q += 1
			return q

		if self.board.Cross[self.board.my][VCF5]:
			self.board.best = list(self.board.Cross[self.board.my][VCF5])[0]
			return 3

		if depth > maxDepth:
			return 0

		# 对方先手且能活四，防守
		if self.board.my != searcher and self.board.Cross[self.board.opp][VCF5]:
			max_q = -1000
			candidates = self.board.getCandidates()
			for candidate in candidates:
				self.board.move(*candidate)
				q = -self.Vct3(searcher, depth + 1, maxDepth, startPoint)
				self.board.delmove()
				if q > 0:
					self.board.best = cand
					return q + 1
				elif q == 0:
					return 0
				elif q > max_q:
					max_q = q
			return max_q

		# 己方先手，且有冲四活三，尝试
		if self.board.my == searcher and self.board.Cross[self.board.my][VCF4]:
			# 对方没有冲四以上，五步胜；否则继续迭代
			if not (self.board.Cross[self.board.opp][VCF5] | self.board.Cross[self.board.opp][VCF4] |
					self.board.Cross[self.board.opp][VCF3] | self.board.Cross[self.board.opp][VCF2] |
					self.board.Cross[self.board.opp][VCF1]):
				self.board.best = list(self.board.Cross[self.board.my][VCF4])[0]
				return 5
			for point in self.board.Cross[self.board.my][VCF4]:
				self.board.move(point[0], point[1])
				q = -self.Vct3(searcher, depth + 1, maxDepth, point)
				self.board.delmove()
				if q > 0:
					self.board.best = point
					return q + 1

		# 己方先手，遍历所有冲四的点 除冲四活三
		if self.board.my == searcher and (
				self.board.Cross[self.board.my][VCF3] | self.board.Cross[self.board.my][VCF2]):
			for point in product(range(startPoint[0] - 4, startPoint[0] + 5),
								 range(startPoint[1] - 4, startPoint[1] + 5)):
				if point in (self.board.Cross[self.board.my][VCF3] | self.board.Cross[self.board.my][VCF2]):
					self.board.move(point[0], point[1])
					q = -self.Vct3(searcher, depth + 1, maxDepth, point)
					self.board.delmove()
					if q > 0:
						self.board.best = point
						return q + 1

		# 己方先手，遍历双活三
		if self.board.my == searcher and self.board.Cross[self.board.my][VCT4]:
			if not (self.board.Cross[self.board.opp][VCF5] | self.board.Cross[self.board.opp][VCF4] |
					self.board.Cross[self.board.opp][VCF3] | self.board.Cross[self.board.opp][VCF2] |
					self.board.Cross[self.board.opp][VCF1]):
				self.board.best = list(self.board.Cross[self.board.my][VCT4])[0]
				return 5
			for point in self.board.Cross[self.board.my][VCT4]:
				self.board.move(point[0], point[1])
				q = -self.Vct3(searcher, depth + 1, maxDepth, point)
				self.board.delmove()
				if q > 0:
					self.board.best = point
					return q + 1
		# 遍历其他情况
		if self.board.my == searcher and (
				self.board.Cross[self.board.my][VCT3] | self.board.Cross[self.board.my][VCT2]):
			for point in product(range(startPoint[0] - 4, startPoint[0] + 5),
								 range(startPoint[1] - 4, startPoint[1] + 5)):
				if point in (self.board.Cross[self.board.my][VCT3] | self.board.Cross[self.board.my][VCT2]):
					self.board.move(point[0], point[1])
					q = -self.Vct3(searcher, depth + 1, maxDepth, point)
					self.board.delmove()
					if q > 0:
						self.board.best = point
						return q + 1

		return 0

	# 三轮VCT
	def Vct3(self, searcher, depth, maxDepth, startPoint):
		if self.board.Cross[self.board.my][VCF6]:
			self.board.best = list(self.board.Cross[self.board.my][VCF6])[0]
			return 1
		# 对方有双连五，2步输
		if len(self.board.Cross[self.board.opp][VCF6]) >= 2:
			return -2
		# offend
		if self.board.Cross[self.board.opp][VCF6]:
			oppPoint = list(self.board.Cross[self.board.opp][VCF6])[0]
			self.board.move(oppPoint[0], oppPoint[1])
			q = -self.Vct3(searcher, depth + 1, maxDepth, startPoint)
			self.board.delmove()
			if q < 0:
				q -= 1
			elif q > 0:
				q += 1
			return q
		# 己方活四，三步胜
		if self.board.Cross[self.board.my][VCF5]:
			return 3

		if depth > maxDepth:
			return 0

		# 对方先手且能活四，防守
		if self.board.my != searcher and self.board.Cross[self.board.my][VCF5]:
			max_q = -1000
			candidates = self.board.getCandidates()
			for candidate in candidates:
				self.board.move(*candidate[0])
				q = -self.Vct3(searcher, depth + 1, maxDepth, startPoint)
				self.board.delmove()
				if q > 0:
					return q + 1
				elif q == 0:
					return 0
				elif q > max_q:
					max_q = q
			return max_q

		# 己方先手且有冲四活三，遍历

		if self.board.my == searcher and self.board.Cross[self.board.my][VCF4]:
			# 对方没有冲四以上，五步胜
			if not (self.board.Cross[self.board.opp][VCF5] | self.board.Cross[self.board.opp][VCF4] |
					self.board.Cross[self.board.opp][VCF3] | self.board.Cross[self.board.opp][VCF2] |
					self.board.Cross[self.board.opp][VCF1]):
				return 5

			for point in self.board.Cross[self.board.my][VCF4]:
				self.board.move(point[0], point[1])
				q = -self.Vct3(searcher, depth + 1, maxDepth, point)
				self.board.delmove()
				if q > 0:
					return q + 1

		# 己方先手，遍历所有冲四的点 除冲四活三
		if self.board.my == searcher and (
				self.board.Cross[self.board.my][VCF3] | self.board.Cross[self.board.my][VCF2]):
			for point in product(range(startPoint[0] - 4, startPoint[0] + 5),
								 range(startPoint[1] - 4, startPoint[1] + 5)):
				if point in (self.board.Cross[self.board.my][VCF3] | self.board.Cross[self.board.my][VCF2]):
					self.board.move(point[0], point[1])
					q = -self.Vct3(searcher, depth + 1, maxDepth, point)
					self.board.delmove()
					if q > 0:
						return q + 1

		# 己方先手遍历双活三
		if self.board.my == searcher and self.board.Cross[self.board.my][VCT4]:
			if not (self.board.Cross[self.board.opp][VCF5] | self.board.Cross[self.board.opp][VCF4] |
					self.board.Cross[self.board.opp][VCF3] | self.board.Cross[self.board.opp][VCF2] |
					self.board.Cross[self.board.opp][VCF1]):
				return 5
			for point in self.board.Cross[self.board.my][VCT4]:
				self.board.move(point[0], point[1])
				q = -self.Vct3(searcher, depth + 1, maxDepth, point)
				self.board.delmove()
				if q > 0:
					return q + 1

		# 遍历其他情况
		if self.board.my == searcher and (
				self.board.Cross[self.board.my][VCT3] | self.board.Cross[self.board.my][VCT2]):
			for point in product(range(startPoint[0] - 4, startPoint[0] + 5),
								 range(startPoint[1] - 4, startPoint[1] + 5)):
				if point in (self.board.Cross[self.board.my][VCT3] | self.board.Cross[self.board.my][VCT2]):
					self.board.move(point[0], point[1])
					q = -self.Vct3(searcher, depth + 1, maxDepth, point)
					self.board.delmove()
					if q > 0:
						return q + 1

		return 0

############## VCT Done ##############
