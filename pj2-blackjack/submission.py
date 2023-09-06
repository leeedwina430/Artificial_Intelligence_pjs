import util, math, random
from collections import defaultdict
from util import ValueIteration

############################################################
# Problem 2a

# If you decide 2a is true, prove it in blackjack.pdf and put "return None" for
# the code blocks below.  If you decide that 2a is false, construct a counterexample.
class CounterexampleMDP(util.MDP):
    # Return a value of any type capturing the start state of the MDP.
    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return 0
        # END_YOUR_CODE

    # Return a list of strings representing actions possible from |state|.
    def actions(self, state):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return ['-1','1']
        # END_YOUR_CODE

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # Remember that if |state| is an end state, you should return an empty list [].
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return [(-1,0.9,-1),(1,0.1,1)] if state == 0 else []
        # END_YOUR_CODE

    # Set the discount factor (float or integer) for your counterexample MDP.
    # 该函数需要返回一个discount，而不是设置
    def discount(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return 1
        # END_YOUR_CODE

############################################################
# Problem 3a

class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: list of integers (face values for each card included in the deck)
        multiplicity: single integer representing the number of cards with each face value
        threshold: maximum number of points (i.e. sum of card values in hand) before going bust
        peekCost: how much it costs to peek at the next card
        """
        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost
        self.printall = False

    # Return the start state.
    # Look closely at this function to see an example of state representation for our Blackjack game.
    # Each state is a tuple with 3 elements:
    #   -- The first element of the tuple is the sum of the cards in the player's hand.
    #   -- If the player's last action was to peek, the second element is the index
    #      (not the face value) of the next card that will be drawn; otherwise, the
    #      second element is None.
    #   -- The third element is a tuple giving counts for each of the cards remaining
    #      in the deck, or None if the deck is empty or the game is over (e.g. when
    #      the user quits or goes bust).
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))   # (N,) represent a tuple

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be placed into the succAndProbReward function below.
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # A few reminders:
    # * Indicate a terminal state (after quitting, busting, or running out of cards)
    #   by setting the deck to None.
    # * If |state| is an end state, you should return an empty list [].
    # * When the probability is 0 for a transition to a particular new state,
    #   don't include that state in the list returned by succAndProbReward.
    '''state : (sumOfCards, PeekIndex, Deck=[N,N,N])'''
    '''return : [ (newState, prob, reward), (newState, prob, reward), ... ]'''
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (our solution is 38 lines of code, but don't worry if you deviate from this)
        sumOfCards, PeekIndex, Deck = state
        if Deck == None:
            return []
        if action == 'Quit':
            return [((sumOfCards,None,None),1,sumOfCards)]

        cardNum = sum(Deck)
        successors = []
        if action == 'Peek':
            if PeekIndex != None:       # peek twice
                return successors
            for ind in range(len(Deck)):
                if Deck[ind]:           # Delete 0 probability cases
                    successors.append(((sumOfCards,ind,Deck), Deck[ind]/cardNum, -self.peekCost))
            return successors

        if action == 'Take':
            if PeekIndex != None:
                new_sum = sumOfCards+self.cardValues[PeekIndex]
                if new_sum > self.threshold:    # go bust
                        new_deck,PeekIndex,reward = None,None,0
                else:
                    new_deck = list(state[2])
                    new_deck[PeekIndex] -= 1
                    new_deck = None if cardNum==1 else tuple(new_deck)
                    reward = 0 if new_deck else new_sum
                return [((new_sum, None, new_deck), 1, reward)]

            for ind in range(len(Deck)):
                if Deck[ind]:          # Delete 0 probability cases
                    new_sum = sumOfCards+self.cardValues[ind]
                    if new_sum > self.threshold:    # go bust
                        new_deck,PeekIndex,reward = None,None,0
                    else:
                        new_deck = list(state[2])
                        new_deck[ind] -= 1
                        new_deck = None if cardNum==1 else tuple(new_deck)
                        reward = 0 if new_deck else new_sum
                    successors.append(((new_sum,None,new_deck), Deck[ind]/cardNum, reward))
            return successors
        # END_YOUR_CODE

    def discount(self):
        return 1

############################################################
# Problem 3b

def peekingMDP():
    """
    Return an instance of BlackjackMDP where peeking is the
    optimal action at least 10% of the time.
    """
    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    return BlackjackMDP([2,3,4,5,20], 2, 20, 1)
    # END_YOUR_CODE

############################################################
# Problem 4a: Q learning

# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearningAlgorithm(util.RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions          # func
        self.discount = discount
        self.featureExtractor = featureExtractor      # func
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)           # weight-a dict:(state,action)
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v        
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1  
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState):
        '''Q^hat(s,a)=w * Φ(s,a); 
           self.getQ(s,a) = self.weights * featureExtractor'''
        # BEGIN_YOUR_CODE (our solution is 9 lines of code, but don't worry if you deviate from this)
        eta = self.getStepSize()
        if newState == None:
            target = reward
        else:
            target = reward + self.discount * max([ self.getQ(newState,naction) for naction in self.actions(newState)])
        prediction = self.getQ(state,action)
        for f,v in self.featureExtractor(state,action):
            self.weights[f] += eta * (target-prediction) * v
        # END_YOUR_CODE

# Return a single-element list containing a binary (indicator) feature      # 判断这种(state,action)对存不存在？？
# for the existence of the (state, action) pair.  Provides no generalization.
def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]

############################################################
# Problem 4b: convergence of Q-learning
# Small test case
smallMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# Large test case
largeMDP = BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3, threshold=40, peekCost=1)

def simulate_QL_over_MDP(mdp, featureExtractor):    # identityFeatureExtractor
    # NOTE: adding more code to this function is totally optional, but it will probably be useful
    # to you as you work to answer question 4b (a written question on this assignment).  We suggest
    # that you add a few lines of code here to run value iteration, simulate Q-learning on the MDP,
    # and then print some stats comparing the policies learned by these two approaches.
    # BEGIN_YOUR_CODE
    rl = QLearningAlgorithm(mdp.actions, mdp.discount(), featureExtractor, explorationProb=0.2)
    util.simulate(mdp, rl, numTrials=30000,maxIterations=1000, verbose=False)
    
    vi = util.ValueIteration()
    vi.solve(mdp)

    rl.explorationProb = 0
    print("RL results v.s. VI results")
    differCount = 0
    if mdp.printall:
        for state in mdp.states:
            rlresult,viresult = rl.getAction(state),vi.pi[state]
            if rlresult != viresult:
                differCount += 1
                print("{}:    RL: {}        VI: {}".format(state,rlresult,viresult))
    else:
        for state in mdp.states:
            if rl.getAction(state) != vi.pi[state] : differCount += 1
    print("Different Actions / Total States: {}/{}    Similarity: {:.2%}".format(differCount,len(mdp.states),1-differCount/len(mdp.states)))
    print("Comparison Done.\n")

    # END_YOUR_CODE
    

############################################################
# Problem 4c: features for Q-learning.

# You should return a list of (feature key, feature value) pairs.
# (See identityFeatureExtractor() above for a simple example.)
# Include the following features in the list you return:
# -- Indicator for the action and the current total (1 feature).
# -- Indicator for the action and the presence/absence of each face value in the deck.
#       Example: if the deck is (3, 4, 0, 2), then your indicator on the presence of each card is (1, 1, 0, 1)
#       Note: only add this feature if the deck is not None.
# -- Indicators for the action and the number of cards remaining with each face value (len(counts) features). ???
#       Note: only add these features if the deck is not None.
def blackjackFeatureExtractor(state, action):
    total, nextCard, counts = state

    # BEGIN_YOUR_CODE (our solution is 7 lines of code, but don't worry if you deviate from this)
    features,feaindi = [ ((action,total), 1) ],[]
    if counts != None:
        for i in range(len(counts)):
            feaindi.append(1 if counts[i] > 0 else 0)
            features.append( ((action,i,counts[i]), 1) )
        features.append( ((action, tuple(feaindi)), 1) )
    return features
    # END_YOUR_CODE

############################################################
# Problem 4d: What happens when the MDP changes underneath you?!

# Original mdp
originalMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# New threshold
newThresholdMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=15, peekCost=1)

def compare_changed_MDP(original_mdp, modified_mdp, featureExtractor):
    # NOTE: as in 4b above, adding more code to this function is completely optional, but we've added
    # this partial function here to help you figure out the answer to 4d (a written question).
    # Consider adding some code here to simulate two different policies over the modified MDP
    # and compare the rewards generated by each.
    # BEGIN_YOUR_CODE
    import numpy as np
    vi = util.ValueIteration()
    vi.solve(original_mdp)
    # vi.solve(modified_mdp)

    rl1 = util.FixedRLAlgorithm(vi.pi)
    rewards1 = util.simulate(modified_mdp, rl1, numTrials=300)
    rl2 = QLearningAlgorithm(original_mdp.actions, original_mdp.discount(), featureExtractor)
    rewards2 = util.simulate(modified_mdp, rl2, numTrials=300)
    print("\n Num Tials = 300")
    print("Expected(average) Reward for Value Iteration: {:.2f}".format(np.mean(rewards1)))         # 关于reward的准确定义????
    print("Expected(average) Reward for Q-learning: {:.2f}".format(np.mean(rewards2)))         # 关于reward的准确定义????

    rewards1 = util.simulate(modified_mdp, rl1, numTrials=30000)
    rewards2 = util.simulate(modified_mdp, rl2, numTrials=20700)
    print("\n Num Tials = 30000")
    print("Expected(average) Reward for Value Iteration: {:.2f}".format(np.mean(rewards1)))         # 关于reward的准确定义????
    print("Expected(average) Reward for Q-learning: {:.2f}".format(np.mean(rewards2)),'\n')         # 关于reward的准确定义????

    # END_YOUR_CODE

