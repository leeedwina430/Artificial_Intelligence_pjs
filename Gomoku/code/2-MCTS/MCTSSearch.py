from utils import Node, State, StringBoard
from math import sqrt, log
from random import choice, uniform
from copy import deepcopy

c = 2


def tree_policy(node):
	"""
	when is_terminal() is True, it means the board is full.
	"""
	while not node.state.is_terminal():
		if node.is_fully_expanded():
			node = best_child(node, c)
		else:
			if uniform(0, 1) < 0.5 and len(node.children) > 0:
				node = best_child(node, c)
			else:
				return expand(node)
	return best_child(node, c)


def expand(node):
	state, move = node.state.next_state()
	new_child = node.add_child(state, move)
	return new_child


def best_child(node, c):
	best_score = float('-inf')
	best_children = []

	for child in node.children:
		# compute UCB score.
		exploitation = child.reward / child.visit_times
		exploration = sqrt(log(node.visit_times) / child.visit_times)

		score = exploitation + c * exploration
		if exploitation > 0:
			score += 5 * exploitation

		if score > best_score:
			best_children = [child]
			best_score = score
		elif score == best_score:
			best_children.append(child)
	return choice(best_children)


def back_propagation(node, reward):
	while node is not None:
		node.update(reward)
		node = node.parent


def UCT_search(budget, state):
	root = Node(state)
	for iteration in range(int(budget)):
		v = tree_policy(root)

		string_board = StringBoard(v.state.player)
		string_board.initial_board(v.state.board)
		reward = string_board.rollout()

		back_propagation(v, reward)

	return best_child(root, c).action


def MCTS_search(board):
	if sum(map(sum, board)) == 0:
		move = (10, 10)
	else:
		root_state = State(deepcopy(board), 1)
		budget = 100
		move = UCT_search(budget, root_state)

	return move
