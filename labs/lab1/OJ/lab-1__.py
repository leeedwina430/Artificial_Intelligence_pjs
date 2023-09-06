import heapq
import sys

class PriorityQueue:

    def __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            assert type(i) == node, 'i must be node'
            if i.state == item.state:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)


class node:
    """define node"""

    def __init__(self, state, parent, path_cost, action,):
        self.state = state
        self.parent = parent
        self.path_cost = path_cost
        self.action = action


class problem:
    """searching problem"""

    def __init__(self, initial_state, actions):
        self.initial_state = initial_state
        self.actions = actions

    def search_actions(self, state):
        """Search actions for the given state.
        Args:
            state: a string e.g. 'A'

        Returns:
            a list of action string list
            e.g. [['A', 'B', '2'], ['A', 'C', '3']]
        """
        ################################# Your code here ###########################
        #actions = []
        #for s in self.actions:
        #    a = s.index(state)
        #    if a == 0:
        #        actions +=  [s]
        #return actions
        #raise Exception	
        return [x for x in self.actions if x[0]==state]

    def solution(self, node):
        """Find the path & the cost from the beginning to the given node.

        Args:
            node: the node class defined above.

        Returns:
            ['Start', 'A', 'B', ....], Cost
        """
        ################################# Your code here ###########################
        b = node
        p = [b.state]
        while b.state != 'Start':
            b = b.parent
            p += b.state
        p.reverse()
        c = node.path_cost
        return p, c
        raise Exception	

    def transition(self, state, action):
        """Find the next state from the state adopting the given action.

        Args:
            state: 'A'
            action: ['A', 'B', '2']

        Returns:
            string, representing the next state, e.g. 'B'
        """
        ################################# Your code here ###########################
        return action[1]
        raise Exception

    def goal_test(self, state):
        """Test if the state is goal

        Args:
            state: string, e.g. 'Goal' or 'A'

        Returns:
            a bool (True or False)
        """
        return state == 'Goal'
        ################################# Your code here ###########################
        raise Exception	

    def step_cost(self, state1, action, state2):
        if (state1 == action[0]) and (state2 == action[1]):
            return int(action[2])
        else:
            print("Step error!")
            sys.exit()

    def child_node(self, node_begin, action):
        """Find the child node from the node adopting the given action

        Args:
            node_begin: the node class defined above.
            action: ['A', 'B', '2']

        Returns:
            a node as defined above
        """
        return node(action[1], node_begin, node_begin.path_cost+self.step_cost(node_begin.state, action, self.transition(node_begin.state, action)), self.search_actions(action[1]))
        ################################# Your code here ###########################
        raise Exception


def UCS(problem):
    """Using Uniform Cost Search to find a solution for the problem.

    Args:
        problem: problem class defined above.

    Returns:
        a list of strings representing the path, along with the path cost as an integer.
            e.g. ['A', 'B', '2'], 5
        if the path does not exist, return 'Unreachable'
    """
    node_test = node(problem.initial_state, '', 0, '')
    frontier = PriorityQueue()
    frontier.push(node_test, node_test.path_cost)
    state2node = {node_test.state: node_test}
    explored = []
    while True:
        if frontier.isEmpty():
            return 'Unreachable', 0
        node_test = frontier.pop()
        if problem.goal_test(node_test.state):
            return problem.solution(node_test)
        if node_test.state not in explored:
            explored += node_test.state
            for i in problem.search_actions(node_test.state):
                child = problem.child_node(node_test, i)
                frontier.update(child, child.path_cost)

    ################################# Your code here ###########################
    raise Exception


if __name__ == '__main__':
    Actions = []
    while True:
        a = input().strip()
        if a != 'END':
            a = a.split()
            Actions += [a]
        else:
            break
    list = [x for x in Actions if x[0]=='Start']
    if len(list) == 0:
        print('Unreachable')

    else:
        graph_problem = problem('Start', Actions)
        answer, path_cost = UCS(graph_problem)
        s = "->"
        if answer == 'Unreachable' and len(list) == 0:
            print('Unreachable')
        else:
            path = s.join(answer)
            print(path)
            print(path_cost)

#%%

def ucs():
    return 1 

a,b = ucs()
print(a)


