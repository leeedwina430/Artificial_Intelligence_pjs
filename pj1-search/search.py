# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

###############################################
# Implementation of Generic Search Algorithms #
###############################################
'''
Here are search functions implemented in generic search algorithm forms.
All functions depend on "generalSearch" function, with different data structures
to represent queues.
'''

def generalSearch(problem, fringe, dfs=True, bfs=False):
    '''
    Generic Search Function
    Parameters "dfs, bfs" are specifically designed for manifesting the possibly 
    relative efficiency of BFS, which means one can check whether the next state 
    is goal state while expanding current state, rather than expanding the states 
    at next depth. This strategy can't be implemented with other algorithms.
    '''
    # Initialization: Add Start State into Fringe
    visited = set()
    fringe.push(([],problem.getStartState(),0))
    if problem.isGoalState(problem.getStartState()):
        return []

    # While Fringe isn't empty, keep expanding new states according to different 
    # data structures of Fringes
    while not fringe.isEmpty():
        cur_actions,cur_state,cur_cost = fringe.pop()
        if cur_state not in visited:         # Double Check
            visited.add(cur_state)      # Graph Search: Prevent repeated visits to the same state
            if dfs and problem.isGoalState(cur_state):  # DFS, UCS, A*: Goal test while expanding the state
                return cur_actions
            
            children = problem.getSuccessors(cur_state) # problem._expended += 1
            for child_state, child_action, child_cost in children:
                if child_state not in visited:   # Graph Search: Check whether we've visited this state
                    child_actions = cur_actions[:]      # Deep copy
                    child_actions.append(child_action)
                    if bfs and problem.isGoalState(child_state):   # BFS: Goal test before the state is expanded
                        return child_actions

                    fringe.push((child_actions,child_state,child_cost+cur_cost))    # Specific to different Fringes
    else:
        return []


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"  
    fringe = util.Stack()
    return generalSearch(problem,fringe)


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    fringe = util.Queue()
    return generalSearch(problem,fringe,False,True)


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    pf = lambda item: problem.getCostOfActions(item[0])
    fringe = util.PriorityQueueWithFunction(pf)
    return generalSearch(problem,fringe)


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # in generalSearch: fringe.push((child_actions,child_state,child_costs))
    pf = lambda item: item[2]+heuristic(item[1],problem)
    fringe = util.PriorityQueueWithFunction(pf)
    return generalSearch(problem,fringe)



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch