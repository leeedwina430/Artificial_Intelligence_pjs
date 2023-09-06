from itertools import product, filterfalse
from mappings import *
from collections import OrderedDict, Counter
from PS import PriorSearch


class Line(object):
	"""
	The representation of a line
	"""

	def __init__(self, Type="row", Index=0, String=""):
		self.type = Type
		self.index = Index
		self.string = String
		self.patterns = [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]
		self.value = [0, 0]  # [player1, player2]

	def update_value(self):
		self.value[0] = sum(map(lambda x: x[0] * x[1], zip(valueTableVector, self.patterns[0])))
		self.value[1] = sum(map(lambda x: x[0] * x[1], zip(valueTableVector, self.patterns[1])))

	def update_pattern(self):
		self.patterns = [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]
		for i in range(4, len(self.string) - 4):
			player = int(self.string[i])
			if player < Empty:
				self.patterns[player][patternDict[self.string[i - 4:i] + self.string[i + 1:i + 5]][player]] += 1


class Board(object):
	"""
	The representation of the board
	"""

	def __init__(self):
		self.height = Height + 8
		self.width = Width + 8
		self.moves = OrderedDict()  # {(x,y):[nShape0,nShape1]},  moves we've done
		self.legalMoves = Counter()  # {(x,y):# moves around}, points that we will consider
		self.row = [Line(Type='row', Index=x, String="0" * self.width) for x in range(self.height)]
		self.col = [Line(Type='col', Index=x, String="0" * self.height) for x in range(self.width)]
		self.diagpri = [Line(Type='diagpri', Index=x, String="0") for x in range(self.height + self.width - 1)]
		self.diagsub = [Line(Type='diagsub', Index=x, String="0") for x in range(self.height + self.width - 1)]
		self.initialBoard()

		self.step = 0
		self.my = Player1
		self.opp = Player2
		self.Cross = [[set() for _ in range(16)] for _ in range(2)]  # next pattern [my=[...],opp=[...]]
		self.best = [4, 4]

	def initialBoard(self):
		"""
		Initial the lines
		"""
		margin = [[Out] * (self.width)] * 4
		self.board = margin + [([Out] * 4) + [Empty for _ in range(self.width - 8)] + ([Out] * 4) for _ in
							   range(self.height - 8)] + margin
		for row_idx in range(self.height):
			str_row = ''.join(map(str, self.board[row_idx]))
			self.row[row_idx].string = str_row

		for col_idx in range(self.width):
			col = [row[col_idx] for row in self.board]
			str_col = ''.join(map(str, col))
			self.col[col_idx].string = str_col

		for i in range(-self.width + 1, self.height):
			if i < 0:
				TL_x, TL_y = (0, -i)
				BR_x, BR_y = (self.width + i, self.width)
			else:
				TL_x, TL_y = (i, 0)
				BR_x, BR_y = (self.height, self.height - i)

			str_diagonal = ''.join(
				map(lambda pos: str(self.board[pos[0]][pos[1]]), zip(range(TL_x, BR_x), range(TL_y, BR_y))))
			self.diagpri[self.width - 1 + TL_x - TL_y].string = str_diagonal

		for i in range(self.width + self.height - 1):
			if i < self.height:
				BL_x, BL_y = (i, 0)
				TR_x, TR_y = (-1, i + 1)
			else:
				BL_x, BL_y = (self.height - 1, i - self.height + 1)
				TR_x, TR_y = (i - self.height - 2, self.width)

			str_diagonal = ''.join(
				map(lambda pos: str(self.board[pos[0]][pos[1]]), zip(range(BL_x, TR_x, -1), range(BL_y, TR_y))))
			self.diagsub[BL_x + BL_y].string = str_diagonal

	def setPoint(self, x, y, piece):
		"""
		update the lines
		"""
		self.row[x].string = self.row[x].string[:y] + str(piece) + self.row[x].string[y + 1:]
		self.col[y].string = self.col[y].string[:x] + str(piece) + self.col[y].string[x + 1:]
		mi = min(x, y)
		self.diagpri[self.width - 1 + x - y].string = self.diagpri[self.width - 1 + x - y].string[:mi] + str(piece) + \
													  self.diagpri[self.width - 1 + x - y].string[mi + 1:]
		mi = min(y, self.height - x - 1)
		self.diagsub[x + y].string = self.diagsub[x + y].string[:mi] + str(piece) + self.diagsub[x + y].string[mi + 1:]

	def getPointPattern(self, x, y):
		"""
		return: patterns[row, col, diag, subdiag], nShape[my, opp]
		"""
		cur_patterns = [[0, 0, 0, 0], [0, 0, 0, 0]]
		# row
		string = self.row[x].string[y - 4:y] + self.row[x].string[y + 1:y + 5]
		cur_patterns[0][0] = patternDict[string][0]
		cur_patterns[1][0] = patternDict[string][1]
		# col
		string = self.col[y].string[x - 4:x] + self.col[y].string[x + 1:x + 5]
		cur_patterns[0][1] = patternDict[string][0]
		cur_patterns[1][1] = patternDict[string][1]
		# pri-diag
		mi = min(x, y)
		string = self.diagpri[self.width - 1 + x - y].string[mi - 4:mi] + self.diagpri[self.width - 1 + x - y].string[
																		  mi + 1:mi + 5]
		cur_patterns[0][2] = patternDict[string][0]
		cur_patterns[1][2] = patternDict[string][1]
		# sub-diag
		mi = min(y, self.height - x - 1)
		string = self.diagsub[x + y].string[mi - 4:mi] + self.diagsub[x + y].string[mi + 1:mi + 5]
		cur_patterns[0][3] = patternDict[string][0]
		cur_patterns[1][3] = patternDict[string][1]
		return cur_patterns, [
			crossTable[cur_patterns[0][0]][cur_patterns[0][1]][cur_patterns[0][2]][cur_patterns[0][3]], \
			crossTable[cur_patterns[1][0]][cur_patterns[1][1]][cur_patterns[1][2]][cur_patterns[1][3]]]

	def updateNbCross(self, x, y, sign):  # sign = +1 / -1
		"""
		update the cross patterns of neighbours of (x,y)
		"""
		# vertical, horizontal, Main diagonal, Sub-diagonal
		dx = (1, 0, 1, 1)
		dy = (0, 1, 1, -1)

		for direction in range(0, 4):
			x_0 = x - 4 * dx[direction]
			y_0 = y - 4 * dy[direction]
			for j in range(0, 9):
				if j == 4: continue
				xx = x_0 + j * dx[direction]
				yy = y_0 + j * dy[direction]
				if self.board[xx][yy] != Empty:
					continue
				_, full = self.getPointPattern(xx, yy)
				if sign == '+':
					self.Cross[0][full[0]].add((xx, yy))
					self.Cross[1][full[1]].add((xx, yy))
				if sign == '-':
					self.Cross[0][full[0]].discard((xx, yy))
					self.Cross[1][full[1]].discard((xx, yy))

	def move(self, x, y, scope=2):
		"""
		put a chess at (x,y)
		"""
		_, prefull = self.getPointPattern(x, y)
		self.Cross[0][prefull[0]].discard((x, y))
		self.Cross[1][prefull[1]].discard((x, y))
		self.moves[(x, y)] = prefull
		self.updateNbCross(x, y, '-')

		# put chess and update information
		self.board[x][y] = self.my
		self.setPoint(x, y, self.my)
		self.updateNbCross(x, y, '+')
		self.step += 1
		self.setplayer(self.opp)

		# update legalMoves
		for xx, yy in product(range(x - scope, x + scope + 1), range(y - scope, y + scope + 1)):
			if self.isFree(xx, yy) and (xx != x or yy != y):
				self.legalMoves[(xx, yy)] += 1

	def delmove(self, scope=2):
		"""
		restore a chess; completed with stack
		"""
		premove, prefull = self.moves.popitem()
		x, y = premove
		self.Cross[0][prefull[0]].add((x, y))
		self.Cross[1][prefull[1]].add((x, y))
		self.updateNbCross(x, y, '-')

		# get off chess and update information
		self.board[x][y] = Empty
		self.setPoint(x, y, Empty)
		self.updateNbCross(x, y, '+')
		self.setplayer(self.opp)
		self.step -= 1

		# update legalMoves
		for xx, yy in product(range(x - scope, x + scope + 1), range(y - scope, y + scope + 1)):
			if self.isFree(xx, yy) and (xx != x or yy != y):
				self.legalMoves[(xx, yy)] -= 1

	def isFree(self, x, y):
		"""
		jugdge whether it's on the board
		"""
		return x >= 4 and y >= 4 and x < self.width - 4 and y < self.height - 4

	def setplayer(self, player):
		"""
		set current player
		"""
		self.my = player
		self.opp = player ^ 1

	def VFA(self):
		"""
		value function approximation: get state estimate value
		"""
		score = [0, 0]
		for lines in [self.row, self.col, self.diagpri, self.diagsub]:
			for line in lines:
				line.update_pattern()
				line.update_value()
				score[0] += line.value[0]
				score[1] += line.value[1]

		return score[self.my] - score[self.opp]

	def getMoveValue(self, x, y):
		"""
		get move estimate value; for pruning
		"""
		cur_patterns = self.getPointPattern(x, y)[0]
		score = [valueTable[cur_patterns[0][0]][cur_patterns[0][1]][cur_patterns[0][2]][cur_patterns[0][3]], \
				 valueTable[cur_patterns[1][0]][cur_patterns[1][1]][cur_patterns[1][2]][cur_patterns[1][3]]]

		if score[self.my] >= 200 or score[self.opp] >= 200:
			if score[self.my] >= score[self.opp]:
				return score[self.my] * 2
			else:
				return score[self.opp]
		return score[self.my] * 2 + score[self.opp]

	def getLegalMoves(self):
		"""
		get legal moves; we only consider points in our neighbour
		"""
		for set in [self.Cross[self.my][VCF6], self.Cross[self.opp][VCF6], self.Cross[self.my][VCF5]]:
			if len(set) > 0: return list(set)

		if len(self.Cross[self.opp][VCF5]) > 0:
			moves_new = []
			for type in [VCF6, VCF5, VCF4, VCF3, VCF2, VCF1]:
				moves_new.extend(list(self.Cross[0][type]))
				moves_new.extend(list(self.Cross[1][type]))
			return moves_new

		moves = Counter()
		for move in [move for move, _ in (self.legalMoves - Counter()).items()]:
			if self.board[move[0]][move[1]] == Empty: moves[move] = self.getMoveValue(move[0], move[1])
		moves = sorted(moves, key=lambda x: moves[x], reverse=True)

		return moves

	def minMax(self, depth: int, alpha: float, beta: float):
		"""
		do min max search
		"""
		if depth == MAXITERATION:
			return self.VFA(), [4, 4]
		legal_moves = self.getLegalMoves()
		if len(legal_moves) == 0:
			return 0, [4, 4]
		legal_moves = legal_moves[0:min(len(legal_moves), BRANCHFACTOR)]
		v = float("-inf")
		max_value, best = 0, [4, 4]
		for move in legal_moves:
			self.move(move[0], move[1])
			value, _ = self.minMax(depth + 1, -beta, -alpha)
			self.delmove()
			if -value > v:
				best = move
				v = -value
				max_value = v
			alpha = max(alpha, v)
			if v >= beta:
				return max_value, best
		return max_value, best

	def rootSearch(self, ps: PriorSearch, depth: int, alpha: float, beta: float):
		"""
		searching from the root
		"""
		legal_moves = self.getLegalMoves()
		if len(legal_moves) == 0:
			return (4, 4)
		legal_moves = legal_moves[0:min(len(legal_moves), BRANCHFACTOR)]

		# defend opponent's
		bestPoint = [legal_moves[0][0], legal_moves[0][1]]
		Lose = True
		max_lose_step = 0
		deletels = []
		for move in legal_moves:
			self.move(move[0], move[1])
			threatPoint = ps.VcfStart()
			if threatPoint != -1:
				result = ps.Vct3(self.my, 0, 10, threatPoint)
				if result > 0:
					deletels.append(move)
					if Lose and (result > max_lose_step):
						max_lose_step = result
						bestPoint = move
				else:
					if Lose:
						Lose = False
						bestPoint = move
			self.delmove()

			# avoid opponent's
		legal_moves = [move for move in legal_moves if (move not in deletels)]
		n = len(legal_moves)
		if n == 0:
			return bestPoint
		if n == 1:
			return legal_moves[0]

		# search for best move
		v = float("-inf")
		best = (4, 4)
		for move in legal_moves:
			self.move(move[0], move[1])
			value, _ = self.minMax(depth + 1, -beta, -alpha)
			self.delmove()
			if -value > v:
				best = move
				v = -value
			alpha = max(alpha, v)
			if v >= beta: return best

		return best

	def Search(self):
		"""
		We will do our Prior Search first then do use min max search
		"""
		# starting point
		if self.step == 0:
			return self.width // 2, self.height // 2

		# find if there exsits ps points
		ps = PriorSearch(self)
		if ps.Vcf1() > 0:
			return self.best
		if ps.Vct1() > 0:
			return self.best

		# further finding
		return self.rootSearch(ps, 0, float("-inf"), float("inf"))
