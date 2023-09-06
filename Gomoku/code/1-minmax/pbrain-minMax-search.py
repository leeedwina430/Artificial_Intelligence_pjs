import random
import pisqpipe as pp
from pisqpipe import DEBUG_EVAL, DEBUG
import re
from collections import Counter
from copy import deepcopy
from itertools import product, filterfalse

pp.infotext = 'name="pbrain-pyrandom", author="Jan Stransky", version="1.0", country="Czech Republic", ' \
			  'www="https://github.com/stranskyjan/pbrain-pyrandom"'

MAX_BOARD = 20
board = [[0 for i in range(MAX_BOARD)] for j in range(MAX_BOARD)]


class BoardScore(object):
	"""
	The representation of the board.
	"""
	__slots__ = ['row', 'col', 'diag_1', 'diag_2', 'score']

	def __init__(self):
		self.row = dict()
		self.col = dict()
		self.diag_1 = dict()
		self.diag_2 = dict()
		self.score = 0

	def initial_board(self, board, player):
		"""
		Convert the board into our representation, which are strings and corresponding values.
		"""
		for row_idx in range(pp.height):
			str_row = ''.join(map(str, board[row_idx]))
			self.row[(player, row_idx)] = [str_row, self.calculate_score(str_row, player)]

		for col_idx in range(pp.width):
			col = [row[col_idx] for row in board]
			str_col = ''.join(map(str, col))
			self.col[(player, col_idx)] = [str_col, self.calculate_score(str_col, player)]

		for i in range(-pp.width + 1, pp.height):
			if i < 0:
				TL_x, TL_y = (0, -i)
				BR_x, BR_y = (pp.width + i, pp.width)
			else:
				TL_x, TL_y = (i, 0)
				BR_x, BR_y = (pp.height, pp.height - i)

			str_diagonal = ''.join(map(lambda pos: str(board[pos[0]][pos[1]]), zip(range(TL_x, BR_x), range(TL_y, BR_y))))
			self.diag_1[(player, pp.width - 1 + TL_x - TL_y)] = [str_diagonal, self.calculate_score(str_diagonal, player)]

		for i in range(pp.width + pp.height - 1):
			if i < pp.height:
				BL_x, BL_y = (i, 0)
				TR_x, TR_y = (-1, i + 1)
			else:
				BL_x, BL_y = (pp.height - 1, i - pp.height + 1)
				TR_x, TR_y = (i - pp.height - 2, pp.width)

			str_diagonal = ''.join(map(lambda pos: str(board[pos[0]][pos[1]]), zip(range(BL_x, TR_x, -1), range(BL_y, TR_y))))
			self.diag_2[(player, BL_x + BL_y)] = [str_diagonal, self.calculate_score(str_diagonal, player)]

	def calculate_score(self, string, player):
		"""
		Given a string and calculate the corresponding values.
		"""
		counter = Counter()
		score = 0

		if player == 1:
			special_cases = black_special_cases
			score_map = black_score_map
		else:
			special_cases = white_special_cases
			score_map = white_score_map

		for case in special_cases.keys():
			counter[case[0]] += len(re.findall(special_cases[case], string))


		for case, count in counter.items():
			score += score_map[case] * count

		return score

	def add_up_score(self):
		score = 0

		# when player = 1, 3 - 2 * player = 1, and when player = 2, 3 - 2 * player = -1.
		# this means if the player is black, we add up the score, and subtract when current player is whits.
		score += sum(list(map(lambda r: (3 - 2 * r[0]) * self.row[r][1], self.row.keys())))

		score += sum(list(map(lambda c: (3 - 2 * c[0]) * self.col[c][1], self.col.keys())))

		score += sum(list(map(lambda diag: (3 - 2 * diag[0]) * self.diag_1[diag][1], self.diag_1.keys())))

		score += sum(list(map(lambda diag: (3 - 2 * diag[0]) * self.diag_2[diag][1], self.diag_2.keys())))

		self.score = score

	def update_score(self, black_update, white_update, move, basic_score):
		"""
		Used when constructing the search tree.
		When one piece has been put down, there are four lines that need to be updated.

		'basic_score' : the new add-up score after putting down the piece.
		'black_update': ((row, row_score), (col, col_score), (diag_1, diag_1_score), (diag_2, diag_2_score)).
						The details of the calculated can be seen in function 'update_score()' below.
		"""
		x, y = move

		# update the score of the black.
		self.row[(1, x)] = [''.join(black_update[0][0]), black_update[0][1]]
		self.col[(1, y)] = [''.join(black_update[1][0]), black_update[1][1]]
		self.diag_1[(1, pp.width - 1 + x - y)] = [''.join(black_update[2][0]), black_update[2][1]]
		self.diag_2[(1, x + y)] = [''.join(black_update[3][0]), black_update[3][1]]

		# update the score of the whits.
		self.row[(2, x)] = [''.join(white_update[0][0]), white_update[0][1]]
		self.col[(2, y)] = [''.join(white_update[1][0]), white_update[1][1]]
		self.diag_1[(2, pp.width - 1 + x - y)] = [''.join(white_update[2][0]), white_update[2][1]]
		self.diag_2[(2, x + y)] = [''.join(white_update[3][0]), white_update[3][1]]

		self.score = basic_score


def brain_init():
	if pp.width < 5 or pp.height < 5:
		pp.pipeOut("ERROR size of the board")
		return
	if pp.width > MAX_BOARD or pp.height > MAX_BOARD:
		pp.pipeOut("ERROR Maximal board size is {}".format(MAX_BOARD))
		return
	pp.pipeOut("OK")

def brain_restart():
	for x in range(pp.width):
		for y in range(pp.height):
			board[x][y] = 0
	pp.pipeOut("OK")


def isFree(x, y):
	return 0 <= y < pp.height and 0 <= x < pp.width and board[x][y] == 0


def brain_my(x, y):
	if isFree(x, y):
		board[x][y] = 1
	else:
		pp.pipeOut("ERROR my move [{},{}]".format(x, y))


def brain_opponents(x, y):
	if isFree(x, y):
		# global opponent_x, opponent_y
		# opponent_x, opponent_y = x, y
		board[x][y] = 2

		# _, basic_score, black_update, white_update = get_score(global_board_score, 2, (x, y))
		# global_board_score.update_score(black_update, white_update, (x, y), basic_score)
	else:
		pp.pipeOut("ERROR opponents's move [{},{}]".format(x, y))


def brain_block(x, y):
	if isFree(x, y):
		board[x][y] = 3
	else:
		pp.pipeOut("ERROR winning move [{},{}]".format(x, y))


def brain_takeback(x, y):
	if x >= 0 and y >= 0 and x < pp.width and y < pp.height and board[x][y] != 0:
		board[x][y] = 0
		return 0
	return 2


def brain_end():
	pass


def brain_about():
	pp.pipeOut(pp.infotext)


if DEBUG_EVAL:
	import win32gui


	def brain_eval(x, y):
		wnd = win32gui.GetForegroundWindow()
		dc = win32gui.GetDC(wnd)
		rc = win32gui.GetClientRect(wnd)
		c = str(board[x][y])
		win32gui.ExtTextOut(dc, rc[2] - 15, 3, 0, None, c, ())
		win32gui.ReleaseDC(wnd, dc)


# BEGIN_OF_MY_NODE

# some special cases that will be used when
black_special_cases = {("WIN", 0): "11111",
					 ("H4", 0): "011110",
					 ("C4", 0): "011112",
					 ("C4", 1): "211110",
					 ("C4", 2): r"^11110",
					 ("C4", 3): r"01111$",
					 ("C4", 4): "10111",
					 ("C4", 5): "11101",
					 ("C4", 6): "11011",
					 ("H3", 0): "01110",
					 ("H3", 1): "010110",
					 ("H3", 2): "011010",

					 ("M3", 0): "001112",
					 ("M3", 1): r"00111$",
					 ("M3", 2): "211100",
					 ("M3", 3): r"^11100",
					 ("M3", 4): "010112",
					 ("M3", 5): r"01011$",
					 ("M3", 6): "211010",
					 ("M3", 7): r"^11010",
					 ("M3", 8): "011012",
					 ("M3", 9): r"01101$",
					 ("M3", 10): "210110",
					 ("M3", 11): r"^10110",
					 ("M3", 12): "10011",
					 ("M3", 13): "11001",
					 ("M3", 14): "10101",
					 ("M3", 15): "2011102",
					 ("M3", 16): r"^011102",
					 ("M3", 17): r"201110$",

					 ("H2", 0): "00110",
					 ("H2", 1): "01100",
					 ("H2", 2): "01010",
					 ("H2", 3): "010010",
					 }

white_special_cases = {("WIN", 0): "22222",
					 ("H4", 0): "022220",
					 ("C4", 0): "022221",
					 ("C4", 1): "122220",
					 ("C4", 2): r"^22220",
					 ("C4", 3): r"02222$",
					 ("C4", 4): "20222",
					 ("C4", 5): "22202",
					 ("C4", 6): "22022",

					 ("H3", 0): "02220",
					 ("H3", 1): "020220",
					 ("H3", 2): "022020",

					 ("M3", 0): "002221",
					 ("M3", 1): r"00222$",
					 ("M3", 2): "122200",
					 ("M3", 3): r"^22200",
					 ("M3", 4): "020221",
					 ("M3", 5): r"02022$",
					 ("M3", 6): "122020",
					 ("M3", 7): r"^22020",
					 ("M3", 8): "022021",
					 ("M3", 9): r"02202$",
					 ("M3", 10): "120220",
					 ("M3", 11): r"^20220",
					 ("M3", 12): "20022",
					 ("M3", 13): "22002",
					 ("M3", 14): "20202",
					 ("M3", 15): "1022201",
					 ("M3", 16): r"^022201",
					 ("M3", 17): r"102220$",

					 ("H2", 0): "00220",
					 ("H2", 1): "02200",
					 ("H2", 2): "02020",
					 ("H2", 3): "020020",
					 }

black_cross_cases = {("WIN", 0): "11111",
					 ("H4", 0): "011110",
					 ("C4", 0): "011112",
					 ("C4", 1): "211110",
					 ("C4", 2): r"^11110",
					 ("C4", 3): r"01111$",
					 ("C4", 4): "10111",
					 ("C4", 5): "11101",
					 ("C4", 6): "11011",
					 ("H3", 0): "01110",
					 ("H3", 1): "010110",
					 ("H3", 2): "011010",
					 }

white_cross_cases = {("WIN", 0): "22222",
					 ("H4", 0): "022220",
					 ("C4", 0): "022221",
					 ("C4", 1): "122220",
					 ("C4", 2): r"^22220",
					 ("C4", 3): r"02222$",
					 ("C4", 4): "20222",
					 ("C4", 5): "22202",
					 ("C4", 6): "22022",
					 ("H3", 0): "02220",
					 ("H3", 1): "020220",
					 ("H3", 2): "022020",
					 }

black_score_map = {"WIN": 200000,
				 "H4": 10000,
				 "C4": 1000,
				 "H3": 200,
				 "M3": 50,
				 "H2": 5,

				 'bonus': 2500}

k = 30  # a weight to emphasis the importance of white pieces.
white_score_map = {"WIN": 200000 * k,
				 "H4": 10000 * k,
				 "C4": 1000 * k,
				 "H3": 20 * k,
				 "M3": 50,
				 "H2": 5,

				 'bonus': 2500}


class Node(object):
	__slots__ = ['successors', 'isLeaf', 'value', 'action']

	def __init__(self, successors=None, isLeaf=False, value=None, action=None):
		self.successors = successors
		self.isLeaf = isLeaf
		self.value = value
		self.action = action  # has the form: (x, y), which is the position that has just been put down a piece.


def max_value(node, alpha, beta):
	"""
	The max-value function is alpha-beta pruning.
	"""
	if node.isLeaf:
		return node.value
	else:
		v = float('-inf')
		for successor in node.successors:
			v = max(v, min_value(successor, alpha, beta))
			alpha = max(alpha, v)
			if v >= beta:
				return v
		return v


def min_value(node, alpha, beta):
	"""
	The min-value function in alpha-beta pruning.
	"""
	if node.isLeaf:
		return node.value
	else:
		v = float('inf')
		for successor in node.successors:
			v = min(v, max_value(successor, alpha, beta))
			beta = min(beta, v)
			if v <= alpha:
				return v
		return v


def constructTree(depth, board_score, player=1, action=None, legal_moves=None):
	node = Node(action=action)
	successors = []
	N = 6  # maximum number of nodes to expand.

	if player == 1:
		# if player = 1, it means current layer is a max-layer.
		reverse = True
	else:
		# otherwise, its a min-layer.
		reverse = False

	if depth == 1:
		scores = []
		for move in legal_moves:
			score, basic_score, black_update, white_update = get_score(board_score, player, move)
			scores.append((score, basic_score, black_update, white_update, move))

		# the order of the nodes can influence the efficiency of the alpha-beta pruning.
		scores.sort(reverse=reverse, key=lambda x: x[0])
		for score in scores[: min(N, len(scores))]:
			successors.append(Node(isLeaf=True, value=score[0], action=score[4]))
	else:
		scores = []
		for move in legal_moves:
			score, basic_score, black_update, white_update = get_score(board_score, player, move)
			scores.append((score, basic_score, black_update, white_update, move))

		scores.sort(reverse=reverse, key=lambda x: x[0])
		for score in scores[: min(N, len(scores))]:
			_, basic_score, black_update, white_update, move = score

			new_board_score = deepcopy(board_score)
			new_board_score.update_score(black_update, white_update, move, basic_score)

			successors.append(
				constructTree(depth - 1, new_board_score,  3 - player, move,
							  update_legal_moves(new_board_score, move, legal_moves)))

	node.successors = successors
	return node


def get_legal_moves(board):
	"""
	Only consider the positions that are near the non-empty positions.
	"""
	moves = set()
	scope = 2

	for x, y in filterfalse(lambda pos: board[pos[0]][pos[1]] == 0, product(range(pp.height), range(pp.width))):
		for dx, dy in product(range(-scope, scope + 1), range(-scope, scope + 1)):
			_x, _y = x + dx, y + dy
			if 0 <= _x < pp.height and 0 <= _y < pp.width and board[_x][_y] == 0:
				moves.add((_x, _y))

	return list(moves)


def update_legal_moves(board_score, move, legal_moves):
	x, y = move
	scope = 2
	new_legal_moves = deepcopy(legal_moves)

	for dx, dy in product(range(-scope, scope + 1), range(-scope, scope + 1)):
		_x, _y = x + dx, y + dy
		if 0 <= _y < pp.height and 0 <= _x < pp.width and list(board_score.row[(1, _x)][0])[_y] == '0' and (_x, _y) not in new_legal_moves:
			new_legal_moves.append((_x, _y))

	new_legal_moves.remove(move)

	return new_legal_moves


def get_score(board_score, player, move):
	x, y = move
	bonus = 2500

	score = board_score.score
	score -= board_score.row[(1, x)][1] + board_score.col[(1, y)][1] + \
			 board_score.diag_1[(1, pp.width - 1 + x - y)][1] + board_score.diag_2[(1, x + y)][1]

	score += board_score.row[(2, x)][1] + board_score.col[(2, y)][1] + \
			 board_score.diag_1[(2, pp.width - 1 + x - y)][1] + board_score.diag_2[(2, x + y)][1]

	black_score, black_update = update_score(board_score, black_special_cases, black_score_map, move, player)
	white_score, white_update = update_score(board_score, white_special_cases, white_score_map, move, player)

	extra_score, _ = update_score(board_score, white_special_cases, white_score_map, move, 3 - player)

	score += black_score - white_score + extra_score

	if player == 1:
		bonus_score = 0.5 * bonus * cross_counter(board_score, 1, move)
		bonus_score += 2 * bonus * cross_counter(board_score, 2, move)
	else:
		bonus_score = 0.5 * bonus * cross_counter(board_score, 2, move)
		bonus_score -= 2 * bonus * cross_counter(board_score, 1, move)

	return score + bonus_score, score, black_update, white_update


def update_score(board_score, special_cases, score_map, move, player):
	x, y = move

	def process(string):
		counter = Counter()
		score = 0

		for case in special_cases.keys():
			counter[case[0]] = max(counter[case[0]], len(re.findall(special_cases[case], string)))


		for case, count in counter.items():
			score += score_map[case] * count

		return score

	# scan by row.
	row = list(board_score.row[(player, x)][0])
	row[y] = str(player)
	row = ''.join(row)
	row_score = process(row)

	# scan by column.
	col = list(board_score.col[(player, y)][0])
	col[x] = str(player)
	col = ''.join(col)
	col_score = process(col)

	# scan by diagonal (from top-left to bottom-right).
	diag_1 = list(board_score.diag_1[(player, pp.width - 1 + x - y)][0])
	diag_1[min(x, y)] = str(player)
	diag_1 = ''.join(diag_1)
	diag_1_score = process(diag_1)

	# scan by diagonal (from bottom-left to top-right)
	diag_2 = list(board_score.diag_2[(player, x + y)][0])
	diag_2[min(y, pp.height - x - 1)] = str(player)
	diag_2 = ''.join(diag_2)
	diag_2_score = process(diag_2)

	score = row_score + col_score + diag_1_score + diag_2_score

	return score, ((row, row_score), (col, col_score), (diag_1, diag_1_score), (diag_2, diag_2_score))


def cross_counter(board_score, player, move):
	case_count = Counter()

	if player == 1:
		cross_cases = black_cross_cases
	else:
		cross_cases = white_cross_cases

	lines = get_cross_strings(move, board_score, player)

	for line in lines:
		for case in cross_cases.keys():
			case_count[case[0]] = max(case_count[case[0]], len(re.findall(cross_cases[case], line)))

	# compute the bonus score.
	if case_count['H4'] + case_count['H3'] + case_count['C4'] > 1:
		bonus_count = case_count['H4'] + case_count['H3'] + case_count['C4']
	else:
		bonus_count = 0

	return bonus_count


def prior_move_search(legal_moves, board_score):

	"""
	In some special cases, the action we should do next is unique.
	If we can find them before minMax search, it can avoid some stupid decisions that our AI may made, and decrease the decision time.
	"""
	move_with_lines = []
	for move in legal_moves:
		lines = get_cross_strings(move, board_score)
		if sum(map(lambda s: len(re.findall('11111', s)), lines)):
			return move
		move_with_lines.append((move, lines))

	# judge whether the opponent will win soon.
	prior_cases = {"122221", r"12222$", r"^22221", "22212", "21222", "22122"}

	for case in prior_cases:
		for move, lines in move_with_lines:
			if sum(map(lambda s: len(re.findall(case, s)), lines)):
				return move

	# judge whether we can make a live-four.
	for move, lines in move_with_lines:
		if sum(map(lambda s: len(re.findall('011110', s)), lines)):
			return move

	return None


def get_cross_strings(move, board_score, player=1):
	"""
	move: the position of a possible movement.
	return: a tuple contains four strings, representing four directions, and the length of each string
			is no more than 9.
	"""
	x, y = move

	# scan by row.
	row = list(board_score.row[(1, x)][0])
	row[y] = str(player)
	row = row[max(0, y - 4): min(pp.width, y + 5)]
	row = ''.join(row)

	# scan by column.
	col = list(board_score.col[(1, y)][0])
	col[x] = str(player)
	col = col[max(0, x - 4): min(pp.height, x + 5)]
	col = ''.join(col)

	# scan by diagonal (from top-left to bottom-right).
	idx = min(x, y)
	diag_1 = list(board_score.diag_1[(1, pp.width - 1 + x - y)][0])
	diag_1[min(x, y)] = str(player)
	sub_diag_1 = diag_1[max(idx - 4, 0): min(idx + 5, len(diag_1))]
	diag_1 = ''.join(sub_diag_1)

	# scan by diagonal (from bottom-left to top-right)
	idx = min(y, pp.height - x - 1)
	diag_2 = list(board_score.diag_2[(1, x + y)][0])
	diag_2[min(y, pp.width - x - 1)] = str(player)
	sub_diag_2 = diag_2[max(0, idx - 4): min(idx + 5, len(diag_2))]
	diag_2 = ''.join(sub_diag_2)

	return row, col, diag_1, diag_2


def expand_tree(root: Node):
	# TODO: a function that can inherit previous search tree to decrease computation.
	return root


def brain_turn():
	pp.height, pp.width = len(board), len(board[0])
	if sum(map(sum, board)) == 0:
		move = (10, 10)
	else:
		board_score = BoardScore()
		board_score.initial_board(board, 1)
		board_score.initial_board(board, 2)
		board_score.add_up_score()

		legal_moves = get_legal_moves(board)

		# do the prior search first.
		move = prior_move_search(legal_moves, board_score)

		# if there does not exist some actions that we must take, do the minMax search.
		if move is None:
			root = constructTree(2, board_score, 1, None, legal_moves=legal_moves)
			move = minMax_search(root)

	pp.do_mymove(move[0], move[1])


def minMax_search(root: Node):
	alpha, beta = float('-inf'), float('inf')
	action = None
	v = float('-inf')
	for successor in root.successors:
		successor_v = min_value(successor, alpha, beta)
		if successor_v > v:
			v = successor_v
			action = successor.action

		# update alpha.
		alpha = max(alpha, v)
	return action



# END_OF_MY_CODE


# "overwrites" functions in pisqpipe module
pp.brain_init = brain_init
pp.brain_restart = brain_restart
pp.brain_my = brain_my
pp.brain_opponents = brain_opponents
pp.brain_block = brain_block
pp.brain_takeback = brain_takeback
pp.brain_turn = brain_turn
pp.brain_end = brain_end
pp.brain_about = brain_about
if DEBUG_EVAL:
	pp.brain_eval = brain_eval


def main():
	pp.main()


if __name__ == "__main__":
	main()
