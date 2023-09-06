import pisqpipe as pp
from collections import Counter
import re
from itertools import filterfalse, product
from utils import StringBoard, get_legal_moves, get_cross_strings


def prior_move_search(legal_moves, string_board):
	"""
	In some special cases, the action we should do next is unique.
	If we can find them before minMax search, it can avoid some stupid decisions that our AI may made, and decrease the decision time.
	"""
	move_with_lines = []
	for move in legal_moves:
		lines = get_cross_strings(move, string_board)
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


def PriorSearch(board):
	stringBoard = StringBoard(1)
	stringBoard.initial_board(board)

	legal_moves = get_legal_moves(board)

	move = prior_move_search(legal_moves, stringBoard)
	return move
