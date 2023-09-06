# searchAgents.py
# ---------------
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
This file contains all of the agents that can be selected to control Pacman.  To
select an agent, use the '-p' option when running pacman.py.  Arguments can be
passed to your agent using '-a'.  For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a fn=depthFirstSearch

Commands to invoke other search strategies can be found in the project
description.

Please only change the parts of the file you are asked to.  Look for the lines
that say

"*** YOUR CODE HERE ***"

The parts you fill in start about 3/4 of the way down.  Follow the project
description for details.

Good luck and happy searching!
"""

from tracemalloc import start
from game import Directions
from game import Agent
from game import Actions
import util
import time
import search

class GoWestAgent(Agent):
    "An agent that goes West until it can't."

    def getAction(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP

#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of search.py          #
#######################################################

class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search
    algorithm for a supplied search problem, then returns actions to follow that
    path.

    As a default, this agent runs DFS on a PositionSearchProblem to find
    location (1,1)

    Options for fn include:
      depthFirstSearch or dfs
      breadthFirstSearch or bfs


    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError(fn + ' is not a search function in search.py.')
        func = getattr(search, fn)
        if 'heuristic' not in func.__code__.co_varnames:
            print(('[SearchAgent] using function ' + fn))
            self.searchFunction = func
        else:
            if heuristic in list(globals().keys()):
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError(heuristic + ' is not a function in searchAgents.py or search.py.')
            print(('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic)))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in list(globals().keys()) or not prob.endswith('Problem'):
            raise AttributeError(prob + ' is not a search problem type in SearchAgents.py.')
        self.searchType = globals()[prob]
        print(('[SearchAgent] using problem type ' + prob))

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None: raise Exception("No search function provided for SearchAgent")
        starttime = time.time()
        problem = self.searchType(state) # Makes a new search problem
        self.actions  = self.searchFunction(problem) # Find a path
        totalCost = problem.getCostOfActions(self.actions)
        print(('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime)))
        if '_expanded' in dir(problem): print(('Search nodes expanded: %d' % problem._expanded))

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP

class PositionSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print('Warning: this does not look like a regular search maze')

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

class StayEastSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the West side of the board.

    The cost function for stepping into a position (x,y) is 1/2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: .5 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn, (1, 1), None, False)

class StayWestSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the East side of the board.

    The cost function for stepping into a position (x,y) is 2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: 2 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)

def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################

class CornersProblem(search.SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function
    """

    def __init__(self, startingGameState):
        """
        Stores the walls, pacman's starting position and corners.
        """
        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top, right = self.walls.height-2, self.walls.width-2
        self.corners = ((1,1), (1,top), (right, 1), (right, top))   # tuple
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                print('Warning: no food in corner ' + str(corner))
        self._expanded = 0 # DO NOT CHANGE; Number of search nodes expanded
        # Please add any code here which you would like to use
        # in initializing the problem
        "*** YOUR CODE HERE ***"
        '''
        state - (Tuple) ((x-axis,y-axis),corner1,corner2,corner3,corner4,corneri...)
        corneri - (Boolean) whether the ith corner has been visited
        '''
        startList = [self.startingPosition]
        startList.extend([False]*len(self.corners))
        self.startingGameState = startingGameState
        self.heuristicInfo = {} # A dictionary for the heuristic to store information
        self.startState = tuple(startList)


    def getStartState(self):
        """
        Returns the start state (in your state space, not the full Pacman state
        space)
        """
        "*** YOUR CODE HERE ***"
        # Update self.startState each time called
        startList = [self.startingPosition]
        startList.extend([False]*len(self.corners))
        self.startState = tuple(startList)
        return self.startState

    def isGoalState(self, state):
        """
        Returns whether this search state is a goal state of the problem.
        """
        "*** YOUR CODE HERE ***"
        # Justify if it is a goalState based on corners' information
        if sum(state[1:]) == len(self.corners):
            return True

        return False


    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
            For a given state, this should return a list of triples, (successor,
            action, stepCost), where 'successor' is a successor to the current
            state, 'action' is the action required to get there, and 'stepCost'
            is the incremental cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            # Add a successor state to the successor list if the action is legal
            # Here's a code snippet for figuring out whether a new position hits a wall:
            #   x,y = currentPosition
            #   dx, dy = Actions.directionToVector(action)
            #   nextx, nexty = int(x + dx), int(y + dy)
            #   hitsWall = self.walls[nextx][nexty]

            "*** YOUR CODE HERE ***"
            # Return next state based on different corners' information
            x,y = state[0]
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            hitsWall = self.walls[nextx][nexty]
            if not hitsWall:
                visited_corners = list(state[1:])
                nextPosition = (nextx,nexty)
                if nextPosition in self.corners:
                    visited_corners[self.corners.index(nextPosition)] = True

                nextState = [nextPosition]
                nextState.extend(visited_corners)
                successors.append( ( tuple(nextState), action, 1) )

        self._expanded += 1 # DO NOT CHANGE
        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999.  This is implemented for you.
        """
        if actions == None: return 999999
        x,y= self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
        return len(actions)     # cost = No. of steps

def cornersHeuristic(state, problem):
    """
    A heuristic for the CornersProblem that you defined.

      state:   The current search state
               (a data structure you chose in your search problem)

      problem: The CornersProblem instance for this layout.

    This function should always return a number that is a lower bound on the
    shortest path from the state to a goal of the problem; i.e.  it should be
    admissible (as well as consistent).
    """
    corners = problem.corners # These are the corner coordinates
    walls = problem.walls # These are the walls of the maze, as a Grid (game.py)

    "*** YOUR CODE HERE ***"

    ''' 
    Method 1: Return maximum position2food manHattan distance
    Result: mediumCorners - Total Cost 106, Expanded 1136, Score 434, Time 0.0s 
            bigCorners    - Total Cost 162, Expanded 4380, Score 378, Time 0.0s
    '''
    f = []
    for i in range(1,5):
        if state[i] == False:
            xy1 = state[0]
            xy2 = problem.corners[i-1]
            f.append(abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])) 
            #f += abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])
            #f += ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5
    
    if len(f) != 0:
        return max(f)
    else:
        return 0


    ''' 
    Method 2: Return (maximum food2food + minimum position2food)'s real path cost
        (When used in Q7: return sum of food2food + minimum position2food real cost)
        (Inspired by Q7 and implemented for Q7-MediumMaze FoodSearchProblem)
    Result: mediumCorners - Total Cost 106, Expanded 383, Score 434, Time 0.7s 
            bigCorners    - Total Cost 162, Expanded 823, Score 378, Time 5.8s 
    '''
    # position, visited_corners = state[0],state[1:]
    # import numpy as np
    # visited = np.array(visited_corners)
    # unvisited_index = np.where(visited==False)[0]

    # def getShortestPathLen(Start, End, problem):
    #     ''' 
    #     Define a function which will be used later
    #     Return the real distance from Start to End on a particular maze with BFS
    #     Save the result in problem.heuristicInfo as well
    #     '''
    #     if problem.heuristicInfo.setdefault((Start,End),None) == None:
    #         subProblem = PositionSearchProblem(problem.startingGameState, goal=End, start=Start,warn=False,visualize=False)
    #         problem.heuristicInfo[(Start,End)] = len(search.breadthFirstSearch(subProblem))
    #     return problem.heuristicInfo[(Start,End)]
        
    # f2fDis,p2fDis = [],[]
    # if len(unvisited_index) == 0:
    #     return 0
    # if len(unvisited_index) == 1:
    #     return getShortestPathLen(position, problem.corners[visited_corners.index(False)], problem)
    
    # for i in range(len(unvisited_index)):
    #     p2fDis.append(getShortestPathLen(position, problem.corners[unvisited_index[i]], problem))
    #     for j in range(i):
    #         f2fDis.append(getShortestPathLen(problem.corners[unvisited_index[i]], problem.corners[unvisited_index[j]], problem))
    
    # # if f2fDis and p2fDis:
    # #     if len(problem.corners) == 4:
    # #         return max(f2fDis) + min(p2fDis)
    # #     else:
    # #         return sum(f2fDis) + min(p2fDis)
    # # else:
    # #     return 0

    # if f2fDis and p2fDis:
    #     return max(f2fDis) + min(p2fDis)
    # else:
    #     return 0


    ''' 
    Method 3: Return (maximum food2food + minimum position2 these 2 food)'s real path cost
        (Inspired by Q7 and implemented for Q7-MediumMaze FoodSearchProblem)
    Result: mediumCorners - Total Cost 106, Expanded 365, Score 434, Time 0.5s 
            bigCorners    - Total Cost 162, Expanded 762, Score 378, Time 3.2s
    '''
    # import numpy as np
    # position, visited_corners = state[0],state[1:]
    # visited = np.array(visited_corners)
    # unvisited_index = np.where(visited==False)[0]
    #
    # def getShortestPathLen(Start, End, problem):
    #     if problem.heuristicInfo.setdefault((Start,End),None) == None:
    #         subProblem = PositionSearchProblem(problem.startingGameState, goal=End, start=Start,warn=False,visualize=False)
    #         problem.heuristicInfo[(Start,End)] = len(search.breadthFirstSearch(subProblem))
    #     return problem.heuristicInfo[(Start,End)]
    #
    # if len(unvisited_index) == 0:
    #     return 0
    # if len(unvisited_index) == 1:
    #     return getShortestPathLen(position, problem.corners[visited_corners.index(False)], problem)
    #
    # diam,f1,f2 = 0,0,0
    # for i in range(len(unvisited_index)):
    #     for ii in range(i):
    #         cur_f2fDis = getShortestPathLen(problem.corners[unvisited_index[i]], problem.corners[unvisited_index[ii]], problem)
    #         if diam < cur_f2fDis:
    #             diam = cur_f2fDis
    #             f1,f2 = i,ii
    #
    # dis = min(getShortestPathLen(position, problem.corners[unvisited_index[f1]], problem),
    #     getShortestPathLen(position, problem.corners[unvisited_index[f2]], problem) )
    # return diam + dis  # max(f2f) + min(p2 these 2 food)


class AStarCornersAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, cornersHeuristic)
        self.searchType = CornersProblem

class FoodSearchProblem:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """
    def __init__(self, startingGameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0 # DO NOT CHANGE
        self.heuristicInfo = {} # A dictionary for the heuristic to store information

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0    

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1 # DO NOT CHANGE
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                successors.append( ( ((nextx, nexty), nextFood), direction, 1) )
        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x,y= self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost

class AStarFoodSearchAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
        self.searchType = FoodSearchProblem


def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    
    position, foodGrid = state
    "*** YOUR CODE HERE ***"
    # Define a function which will be used later
    # Return the real distance from Start to End on a particular maze
    # Save the result in problem.heuristicInfo as well
    def getShortestPathLen(Start, End, problem):
        if problem.heuristicInfo.setdefault((Start,End),None) == None:
            subProblem = PositionSearchProblem(problem.startingGameState, goal=End, start=Start,warn=False,visualize=False)
            problem.heuristicInfo[(Start,End)] = len(search.aStarSearch(subProblem))
        return problem.heuristicInfo[(Start,End)]
    
    # This variant will be used in every algorithm later
    foodList = foodGrid.asList()

    '''
    Method 0: Max manhattan distance from current position to foods
    Result: TinySearch   - Total Cost 27, Expanded 2468, Score 573, Time 0.2s 
            TrickySearch - Total Cost 60, Expanded 9551, Score 570, Time 1.3s 
            MediumSearch - Fails
    '''
    # for i in range(len(foodList)):
    #     xy = foodList[i]
    #     foodList[i] = (abs(xy[0] - position[0]) + abs(xy[1] - position[1])) 
    
    # if foodList:
    #     return max(foodList)
    # else:
    #     return 0


    '''
    Method 1: Max real distance from current position to foods
    Result: TinySearch   - Total Cost 27, Expanded 2372, Score 573, Time 0.2s 
            TrickySearch - Total Cost 60, Expanded 4137, Score 570, Time 0.7s 
            MediumSearch - Fails
    '''
    # for i in range(len(foodList)):
    #     food = foodList[i]
    #     if problem.heuristicInfo.setdefault((position,food),None) == None:
    #         subProblem = PositionSearchProblem(problem.startingGameState, goal=food, start=position,warn=False,visualize=False)
    #         foodList[i] = len(search.aStarSearch(subProblem, heuristic=manhattanHeuristic)) # search 返回actions
    # #        foodList[i] = len(search.uniformCostSearch(subProblem))
    # #        foodList[i] = len(search.depthFirstSearch(subProblem))
    # #        foodList[i] = len(search.breadthFirstSearch(subProblem))
    #         problem.heuristicInfo[(position,food)] = foodList[i]
    #     else:
    #         foodList[i] = problem.heuristicInfo[(position,food)]
    #
    # if foodList:
    #     return max(foodList)
    # else:
    #     return 0


    '''
    Method 2: Maximum food2food distance + Minimum position2food distance 
    Result: TinySearch   - Total Cost 27, Expanded 911, Score 573, Time 0.1s 
            TrickySearch - Total Cost 60, Expanded 719, Score 570, Time 0.3s 
            MediumSearch - Fails
    '''
    # f2fDis,p2fDis = [],[]

    # if len(foodList) == 1:
    #     return getShortestPathLen(position, foodList[0], problem)

    # for i in range(len(foodList)):
    #     p2fDis.append(getShortestPathLen(position, foodList[i], problem))
    #     for j in range(i):
    #         f2fDis.append(getShortestPathLen(foodList[i], foodList[j], problem))
            
    # if f2fDis and p2fDis:
    #    return max(f2fDis) + min(p2fDis)
    # else:
    #    return 0


    '''
    Method 2.5: Sum of food2food distance + Minimum position2food distance 
    Result: TinySearch   - Total Cost 31, Expanded 33, Score 569, Time 0.0s 
            TrickySearch - Total Cost 68, Expanded 77, Score 562, Time 0.1s
            MediumSearch - Total Cost 159, Expanded 159, Score 1421, Time 2.9s
            bigSearch    - Total Cost 292, Expanded 299, Score 2418, Time 48.8s
    Notice: Inadmissible!
    '''
    # if f2fDis and p2fDis:
    #    return sum(f2fDis) + min(p2fDis)
    # else:
    #    return 0


    '''
    Method 3: Maximum food2food + Minimum food2food + Minimum position2food distance 
    Result: TinySearch   - Total Cost 27, Expanded 623, Score 573, Time 0.1s 
            TrickySearch - Total Cost 60, Expanded 168, Score 570, Time 0.1s 
            MediumSearch - Fails
    Notice: Inadmissible if the problem size is bigger.
    '''
    # f2fDis,p2fDis = [],[]
    
    # if len(foodList) == 1:
    #     return getShortestPathLen(position, foodList[0], problem)
    
    # for i in range(len(foodList)):
    #     p2fDis.append(getShortestPathLen(position, foodList[i], problem))
    #     for j in range(i):
    #         f2fDis.append(getShortestPathLen(foodList[i], foodList[j], problem))
          
    # f2fDis_re = sorted(f2fDis,reverse=True)
    # #print(len(f2fDis))
    # result = min(p2fDis) if p2fDis else 0
    # result += f2fDis_re[0] + f2fDis[0] if len(f2fDis)>6 else (f2fDis_re[0] if len(f2fDis) != 0 else 0)
    # # len(f2fDis) > 5 fails
    # return result   # max(f2fDis) + min(f2fDis) + min(p2fDis)


    '''
    Method 4: Maximum food2food distance + smaller position2food distance(for these 2 food)
    Result: TinySearch   - Total Cost 27, Expanded 532, Score 573, Time 0.1s 
            TrickySearch - Total Cost 60, Expanded 376, Score 570, Time 0.1s 
            MediumSearch - Fails
    '''
    foodCount = len(foodList)
    if foodCount == 1:
        return getShortestPathLen(position, foodList[0], problem)
    if foodCount == 0:
        return 0
    # Find the 'diam' of the points and the min distance to one of the end points of the diam
    diam,f1,f2 = 0,0,0
    for i in range(foodCount):
        for ii in range(i):
            cur_f2fDis = getShortestPathLen(foodList[i], foodList[ii], problem)
            if diam < cur_f2fDis:
                diam = cur_f2fDis
                f1,f2 = i,ii
    
    dis = min(getShortestPathLen(position, foodList[f1], problem),
        getShortestPathLen(position, foodList[f2], problem) )
    return diam + dis  # max(f2f) + min(p2f)


    '''
    Method 5: (Designed for medium and big Mazes particularly)
              Select some "Tricky Points(TPs)" which is defined as the points that have 3 walls
              around it. 
    Result: TinySearch   - Total Cost 31, Expanded 45, Score 569, Time 0.0s 
            TrickySearch - Total Cost 60, Expanded 376, Score 570, Time 0.1s 
            MediumSearch - Total Cost 157, Expanded 34846, Score 1423, Time 27.2s
    Notice: Inadmissible!
    '''
    # # Define a function which will be used later
    # # Return the cost of shortest path from one tricky point to all tricky points
    # # Save the result in problem.heuristicInfo as well
    # def getTP2TPsPathLen(Position,problem,Corners):
    #     if problem.heuristicInfo.setdefault('subState',None) == None:
    #         problem.heuristicInfo['subState'] = problem.startingGameState.deepCopy()
    #         top, right = problem.heuristicInfo['subState'].getWalls().height-2, problem.heuristicInfo['subState'].getWalls().width-2
    #         for x,y in [(1,1), (1,top), (right, 1), (right, top)]:
    #             problem.heuristicInfo['subState'].data.food[x][y] = True

    #     if problem.heuristicInfo.setdefault((Position,Corners),None) == None:
    #         subProblem = CornersProblem(problem.heuristicInfo['subState'])
    #         subProblem.corners = Corners
    #         subProblem.startingPosition = Position
    #         subProblem.startState = subProblem.getStartState()  # Update Start State
    #         problem.heuristicInfo[(Position,Corners)] = len(search.aStarSearch(subProblem, cornersHeuristic))
    #     return problem.heuristicInfo[(Position,Corners)]

    # if len(foodList) == 0:
    #     return 0
    # if len(foodList) == 1:
    #     return getShortestPathLen(position, foodList[0], problem)

    # trickyPoints = []
    # for food in foodList:
    #     if problem.walls[food[0]+1][food[1]] + problem.walls[food[0]][food[1]+1] + problem.walls[food[0]-1][food[1]] + problem.walls[food[0]][food[1]-1] == 3:
    #         trickyPoints.append(food)

    # if len(trickyPoints) <= 2:
    #     #p2fDis = []
    #     #for i in range(len(foodList)):
    #     #    p2fDis.append(getShortestPathLen(position, foodList[i], problem))
    #     diam,f1,f2 = 0,0,0
    #     for i in range(len(foodList)):
    #         for ii in range(i):
    #             cur_f2fDis = getShortestPathLen(foodList[i], foodList[ii], problem)
    #             if diam < cur_f2fDis:
    #                 diam = cur_f2fDis
    #                 f1,f2 = i,ii
    #     dis = min( getShortestPathLen(position, foodList[f1], problem),
    #         getShortestPathLen(position, foodList[f2], problem) )
    #     return diam + dis
    
    # # # Unfinished implementation
    # # if len(trickyPoints) > 6:
    # #     dis,nearest = 99999,None
    # #     for food in trickyPoints:
    # #         cur_dis = getShortestPathLen(position, food, problem)
    # #         if dis > cur_dis:
    # #             dis = cur_dis
    # #             nearest = food
    # #     return dis + getTP2TPsPathLen(nearest,problem,tuple(trickyPoints))
    # p2fDis,f2fDis = [],[]
    # for i in range(len(foodList)):
    #     p2fDis.append(getShortestPathLen(position, foodList[i], problem))
    #     for j in range(i):
    #         f2fDis.append(getShortestPathLen(foodList[i], foodList[j], problem))
            
    # if f2fDis and p2fDis:
    #    return sum(f2fDis) + min(p2fDis)
    # else:
    #    return 0
    

class ClosestDotSearchAgent(SearchAgent):
    "Search for all food using a sequence of searches"
    def registerInitialState(self, state):
        self.actions = []
        currentState = state
        while(currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState) # The missing piece
            self.actions += nextPathSegment
            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    t = (str(action), str(currentState))
                    raise Exception('findPathToClosestDot returned an illegal move: %s!\n%s' % t)
                currentState = currentState.generateSuccessor(0, action)
        self.actionIndex = 0
        print('Path found with cost %d.' % len(self.actions))

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition()
        food = gameState.getFood()
        #walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState)

        "*** YOUR CODE HERE ***"
        foodList = {}
        for afood in food.asList():
            problem.startState = startPosition
            problem.goal = afood
            #cur_path = search.breadthFirstSearch(problem)
            #cur_path = search.uniformCostSearch(problem)    # if D == 1: (bfs <=> ucs)
            cur_path = search.aStarSearch(problem, heuristic=manhattanHeuristic)
            #cur_path = search.aStarSearch(problem, heuristic=euclideanHeuristic)
            foodList[len(cur_path)] = cur_path
        
        if foodList:
            return foodList[sorted(foodList.keys())[0]]
        else:
            return []


class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.
    # state space and successor function is just like the PositionSearchProblem

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, gameState):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        #x,y = state
        "*** YOUR CODE HERE ***"
        return state == self.goal
