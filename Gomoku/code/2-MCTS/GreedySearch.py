import re
from collections import Counter
from itertools import product, filterfalse
from copy import deepcopy


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
					   }


def cross_counter(board, player, move):
	"""
	如果只考虑当前位置四个方向的话，在判断对手的局势的时候会忽略对手即将赢的摆法。
	board is already a deepcopy one.
	"""
	if player == 1:
		cross_cases = black_special_cases
	else:
		cross_cases = white_special_cases

	height, width = len(board), len(board[0])
	x, y = move
	board[x][y] = player
	case_count = Counter()

	# scan by row.
	row = ''.join(map(str, board[x][max(0, y - 4): min(width, y + 5)]))

	# scan by column.
	col = ''.join(map(str, [row[y] for row in board[max(0, x - 4): min(height, x + 5)]]))

	# scan by diagonal (from top-left to bottom-right).
	TL_x, TL_y = x - min(x, y, 4), y - min(x, y, 4)
	BR_x, BR_y = x + min(height - 1 - x, width - 1 - y, 4), y + min(height - 1 - x, width - 1 - y, 4)
	diag_1 = ''.join(map(lambda pos: str(board[pos[0]][pos[1]]), zip(range(TL_x, BR_x + 1), range(TL_y, BR_y + 1))))

	# scan by diagonal (from bottom-left to top-right)
	BL_x, BL_y = x + min(height - 1 - x, y, 4), y - min(height - 1 - x, y, 4)
	TR_x, TR_y = x - min(x, width - 1 - y, 4), y + min(x, width - 1 - y, 4)
	diag_2 = ''.join(map(lambda pos: str(board[pos[0]][pos[1]]), zip(range(BL_x, TR_x - 1, -1), range(BL_y, TR_y + 1))))

	board[x][y] = 0
	searches = [(row, Counter()), (col, Counter()), (diag_1, Counter()), (diag_2, Counter())]

	def process(search):
		string, count = search
		for case in cross_cases.keys():
			count[case[0]] += len(re.findall(cross_cases[case], string))
		return count

	for result in map(process, searches):
		case_count += result

	if case_count['H4'] + case_count['H3'] + case_count['C4'] > 1:
		case_count['bonus'] = case_count['H4'] + case_count['H3'] + case_count['C4']

	return case_count['bonus']



def GreedySearch(board, player, legal_moves):
	max_score = float('-inf')
	action = None
	for move in legal_moves:
		score = get_score(board, player, move)
		if score > max_score:
			max_score = score
			action = move

	return action


def get_score(board, player, pos):
	score_map = {"WIN": 200000,
				 "H4": 10000,
				 "C4": 1000,
				 "H3": 200,

				 'bonus': 2500}

	new_board = deepcopy(board)
	new_board[pos[0]][pos[1]] = player
	score = 0

	for case, count in special_cases_counter(new_board, player).items():
		score += score_map[case] * count

	score += score_map['bonus'] * cross_counter(new_board, player, pos)

	for case, count in special_cases_counter(new_board, 3 - player).items():
		if case in ['H4', 'C4', 'WIN', 'H3']:
			score -= 30 * score_map[case] * count
		else:
			score -= score_map[case] * count

	# new_board[pos[0]][pos[1]] = player
	score += score_map['bonus'] * cross_counter(new_board, 3 - player, pos)

	return score


def special_cases_counter(board, player):
	if player == 1:
		special_cases = black_special_cases
	else:
		special_cases = white_special_cases

	height, width = len(board), len(board[0])
	case_count = Counter()

	# scan by row.
	for row in board:
		str_row = ''.join(map(str, row))
		for case in special_cases.keys():
			case_count[case[0]] += len(re.findall(special_cases[case], str_row))

	# scan by column.
	for col_idx in range(width):
		col = [row[col_idx] for row in board]
		str_col = ''.join(map(str, col))
		for case in special_cases.keys():
			case_count[case[0]] += len(re.findall(special_cases[case], str_col))

	# scan by diagonal (from top-left to bottom-right).
	for i in range(-width + 1, height):
		if i < 0:
			TL_x, TL_y = (0, -i)
			BR_x, BR_y = (width + i, width)
		else:
			TL_x, TL_y = (i, 0)
			BR_x, BR_y = (height, height - i)

		str_diagonal = ''.join(
			map(lambda pos: str(board[pos[0]][pos[1]]), zip(range(TL_x, BR_x), range(TL_y, BR_y))))
		for case in special_cases.keys():
			case_count[case[0]] += len(re.findall(special_cases[case], str_diagonal))

	# scan by diagonal (from bottom-left to top-right)
	for i in range(width + height - 1):
		if i < height:
			BL_x, BL_y = (i, 0)
			TR_x, TR_y = (-1, i + 1)
		else:
			BL_x, BL_y = (height - 1, i - height + 1)
			TR_x, TR_y = (i - height - 2, width)

		str_diagonal = ''.join(
			map(lambda pos: str(board[pos[0]][pos[1]]), zip(range(BL_x, TR_x, -1), range(BL_y, TR_y))))

		for case in special_cases.keys():
			case_count[case[0]] += len(re.findall(special_cases[case], str_diagonal))

	return case_count
