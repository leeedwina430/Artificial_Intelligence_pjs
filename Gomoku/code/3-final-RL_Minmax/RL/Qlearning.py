from Board import *
import numpy as np
from copy import deepcopy
from tqdm import tqdm

Player1 = 0
Player2 = 1

class Qlearner:
    def __init__(self, network, lr=0.01, gamma=0.999):
        self.network = network
        self.lr = lr
        self.gamma = gamma
        self.C = 10  # fixed para

    def learn(self, numTrails=10):
        fixed_network = deepcopy(self.network)      # fixed 

        for t in tqdm(range(numTrails)):
            board = Board(fixed_network, explore_prob=0.2)
            x,y = board.Search()  # start point
            board.move(x,y)

            step = 1
            while True:     # simulate until the end
                x,y = board.Search()
                
                board.move(x, y)        # place
                feature = board.extract_feature()   # get feature

                winner = -1 if step < 10 else board.is_win()    
                if winner == Player1:
                    target = np.array(1).reshape((1, 1))
                    self.network.train(feature, target)
                elif winner == Player2:
                    target = np.array(0).reshape((1, 1))
                    self.network.train(feature, target)
                else:
                    bestQ, _ = board.Q_value()
                    target = np.array(self.gamma * bestQ).reshape((1, 1))
                self.network.train(feature, target)

                if winner > -1 or board.isFull():
                    break

                step += 1

            # DDQN update
            if (t + 1) % self.C == 0:
                fixed_network = deepcopy(self.network) 

            WD = "./try/"
            print('###########', t)
            print(self.network.params)
            print(board.board)
            np.save(WD+'para.npy', self.network.params)

