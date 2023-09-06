import pisqpipe as pp
from itertools import product, filterfalse
from random import choice
from copy import deepcopy
import re
from GreedySearch import cross_counter

filter_black_cases = {
	"011112",
	"211110",
	"^11110",
	"01111$",
	"10111",
	"11101",
	"11011",
	"01110",
	"010110",
	"011010",
}

filter_white_cases = {
	"12220",
	"02221",
	"021220",
	"022120",
}

filter_better_cases = {
	"12221",
	"01112",
	"21110",
	"010112",
	"211010",
	"011012",
	"210110",
	"10011",
	"11001",
}


def get_cross_strings(move, string_board, player=1):
	"""
	move: the position of a possible movement.
	return: a tuple contains four strings, representing four directions, and the length of each string
			is no more than 9.
	"""
	x, y = move

	# scan by row.
	row = list(string_board.row[x])
	row[y] = str(player)
	row = row[max(0, y - 4): min(pp.width, y + 5)]
	row = ''.join(row)

	# scan by column.
	col = list(string_board.col[y])
	col[x] = str(player)
	col = col[max(0, x - 4): min(pp.height, x + 5)]
	col = ''.join(col)

	# scan by diagonal (from top-left to bottom-right).
	idx = min(x, y)
	diag_1 = list(string_board.diag_1[pp.width - 1 + x - y])
	diag_1[min(x, y)] = str(player)
	sub_diag_1 = diag_1[max(idx - 4, 0): min(idx + 5, len(diag_1))]
	diag_1 = ''.join(sub_diag_1)

	# scan by diagonal (from bottom-left to top-right)
	idx = min(y, pp.height - x - 1)
	diag_2 = list(string_board.diag_2[x + y])
	diag_2[min(y, pp.width - x - 1)] = str(player)
	sub_diag_2 = diag_2[max(0, idx - 4): min(idx + 5, len(diag_2))]
	diag_2 = ''.join(sub_diag_2)

	return row, col, diag_1, diag_2


def get_legal_moves(board, scope=1):
	"""
	Only consider the positions that are near the non-empty positions.
	"""
	moves = set()

	for x, y in filterfalse(lambda pos: board[pos[0]][pos[1]] == 0, product(range(pp.height), range(pp.width))):
		for dx, dy in product(range(-scope, scope + 1), range(-scope, scope + 1)):
			_x, _y = x + dx, y + dy
			if 0 <= _x < pp.height and 0 <= _y < pp.width and board[_x][_y] == 0:
				moves.add((_x, _y))

	return list(moves)


class StringBoard:
	__slots__ = ['board', 'player', 'row', 'col', 'diag_1', 'diag_2', 'possible_moves']

	def __init__(self, player):
		self.player = player
		self.row = dict()
		self.col = dict()
		self.diag_1 = dict()
		self.diag_2 = dict()

	def initial_board(self, board, scope=1):
		"""
		完成两个任务，一个是初始化 possible_moves, 将非空点周围的点加入possible_moves.
		其次，将整个棋盘转化为用字符串表示的形式，一共 20 + 20 + 39 + 39 = 118 个键值对.
		并且会对输入的 board 进行 deepcopy.
		"""
		self.board = deepcopy(board)

		possible_moves = set()
		for x, y in filterfalse(lambda pos: self.board[pos[0]][pos[1]] == 0,
								product(range(pp.height), range(pp.width))):
			for dx, dy in product(range(-scope, scope + 1), range(-scope, scope + 1)):
				_x, _y = x + dx, y + dy
				if 0 <= _x < pp.height and 0 <= _y < pp.width and self.board[_x][_y] == 0:
					possible_moves.add((_x, _y))

		# TODO: 11早上之前用的是下面这几行
		# for x, y in product(range(pp.height), range(pp.width)):
		# 	if 0 <= x < pp.height and 0 <= y < pp.width and self.board[x][y] == 0:
		# 		possible_moves.add((x, y))

		self.possible_moves = list(possible_moves)


		for row_idx in range(pp.height):
			str_row = ''.join(map(str, board[row_idx]))
			self.row[row_idx] = str_row

		for col_idx in range(pp.width):
			col = [row[col_idx] for row in board]
			str_col = ''.join(map(str, col))
			self.col[col_idx] = str_col

		for i in range(-pp.width + 1, pp.height):
			if i < 0:
				TL_x, TL_y = (0, -i)
				BR_x, BR_y = (pp.width + i, pp.width)
			else:
				TL_x, TL_y = (i, 0)
				BR_x, BR_y = (pp.height, pp.height - i)

			str_diagonal = ''.join(
				map(lambda pos: str(board[pos[0]][pos[1]]), zip(range(TL_x, BR_x), range(TL_y, BR_y))))
			self.diag_1[pp.width - 1 + TL_x - TL_y] = str_diagonal

		for i in range(pp.width + pp.height - 1):
			if i < pp.height:
				BL_x, BL_y = (i, 0)
				TR_x, TR_y = (-1, i + 1)
			else:
				BL_x, BL_y = (pp.height - 1, i - pp.height + 1)
				TR_x, TR_y = (i - pp.height - 2, pp.width)

			str_diagonal = ''.join(
				map(lambda pos: str(board[pos[0]][pos[1]]), zip(range(BL_x, TR_x, -1), range(BL_y, TR_y))))
			self.diag_2[BL_x + BL_y] = str_diagonal

	def update(self, move, scope=1, player=None):
		"""
		给定输入的 move, 更改4个字符串.
		"""
		if player is None:
			player = self.player

		x, y = move
		self.board[x][y] = player

		row = list(self.row[x])
		row[y] = str(player)
		self.row[x] = ''.join(row)

		col = list(self.col[y])
		col[x] = str(player)
		self.col[y] = ''.join(col)

		diag_1 = list(self.diag_1[pp.width - 1 + x - y])
		diag_1[min(x, y)] = str(player)
		self.diag_1[pp.width - 1 + x - y] = ''.join(diag_1)

		diag_2 = list(self.diag_2[x + y])
		diag_2[min(y, pp.height - x - 1)] = str(player)
		self.diag_2[x + y] = ''.join(diag_2)

		for dx, dy in product(range(-scope, scope + 1), range(-scope, scope + 1)):
			_x, _y = x + dx, y + dy
			if 0 <= _x < pp.height and 0 <= _y < pp.width and self.board[_x][_y] == 0 and (
					_x, _y) not in self.possible_moves:
				self.possible_moves.append((_x, _y))
		# if move in self.possible_moves:
		# 	self.possible_moves.remove(move)

	def judge_winner(self, move, player=None):
		if player is None:
			player = self.player

		if player == 1:
			win_pattern = "11111"
		else:
			win_pattern = "22222"

		x, y = move
		for line in [self.row[x], self.col[y], self.diag_1[min(x, y)], self.diag_2[min(y, pp.height - x - 1)]]:
			if len(re.findall(win_pattern, line)):
				return True

		return False

	def rollout(self, scope=1):
		while self.possible_moves:
			move = choice(self.possible_moves)
			# move = GreedySearch(self.board, self.player, self.possible_moves)
			self.update(move)

			if self.judge_winner(move, 1):
				return 1
			elif self.judge_winner(move, 2):
				return -1

			self.player = 3 - self.player
		return 0


class State:
	__slots__ = ['board', 'player', 'moves', 'possible_moves']

	def __init__(self, board, player, moves=None, possible_moves=None):
		"""
		moves: records the historical moves.
		"""
		self.board = board
		self.player = player
		if moves is None:
			self.moves = []
		else:
			self.moves = moves
		if possible_moves is None:
			legal_moves = get_legal_moves(self.board)
			self.possible_moves = self.filter_moves(legal_moves)
		else:
			self.possible_moves = possible_moves

	def next_state(self):
		# move = GreedySearch(self.board, self.player, self.possible_moves)
		move = choice(self.possible_moves)
		x, y = move

		new_board = deepcopy(self.board)
		new_board[x][y] = self.player

		self.possible_moves.remove(move)
		new_possible_moves = deepcopy(self.possible_moves)

		# for dx, dy in product(range(-scope, scope + 1), range(-scope, scope + 1)):
		# 	_x, _y = x + dx, y + dy
		# 	if 0 <= _x < pp.height and 0 <= _y < pp.width and self.board[_x][_y] == 0 and (
		# 			_x, _y) not in new_possible_moves:
		# 		new_possible_moves.append((_x, _y))

		new_state = State(new_board, 3 - self.player, self.moves + [move], new_possible_moves)

		return new_state, move

	def is_terminal(self):
		"""
		check whether the game is over, or the depth has reached the maximum.
		return: (winner, whether terminal), winner == 0 means no one wins.
		"""
		if sum(map(sum, self.board)) == 600 or len(self.possible_moves) == 0:
			return True
		else:
			return False

	def filter_moves(self, legal_moves):
		filter_board = StringBoard(1)
		filter_board.initial_board(self.board)
		filter_moves = []

		move_with_lines = []
		for move in legal_moves:
			if cross_counter(self.board, self.player, move):
				filter_moves.append(move)
			elif cross_counter(self.board, 3 - self.player, move):
				filter_moves.append(move)
			else:
				move_with_lines.append([move, get_cross_strings(move, filter_board)])

		for move, lines in move_with_lines:
			for case in filter_white_cases:
				if sum(map(lambda s: len(re.findall(case, s)), lines)):
					filter_moves.append(move)
					break

		if len(filter_moves) > 0:
			return filter_moves

		for move, lines in move_with_lines:
			for case in filter_black_cases:
				if sum(map(lambda s: len(re.findall(case, s)), lines)):
					filter_moves.append(move)
					break

		if len(filter_moves) > 0:
			return filter_moves

		for move, lines in move_with_lines:
			for case in filter_better_cases:
				if sum(map(lambda s: len(re.findall(case, s)), lines)):
					filter_moves.append(move)
					break

		if len(filter_moves) > 0:
			return filter_moves

		return legal_moves


class Node:
	__slots__ = ['state', 'action', 'children', 'parent', 'visit_times', 'reward']

	def __init__(self, state, parent=None, action=None):
		self.state = state
		self.action = action
		self.children = []
		self.parent = parent
		self.visit_times = 0
		self.reward = 0

	def add_child(self, state, move):
		child = Node(state, self, move)
		self.children.append(child)
		return child

	def is_fully_expanded(self):
		if len(self.state.possible_moves) == 0:
			return True
		else:
			return False

	def update(self, reward):
		self.reward += reward
		self.visit_times += 1
