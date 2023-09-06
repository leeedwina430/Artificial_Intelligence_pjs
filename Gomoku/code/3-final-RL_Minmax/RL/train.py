import Qlearning
import NeuralNetwork as nn
import numpy as np

WD = "./try/"
network = nn.NN()
learner = Qlearning.Qlearner(network)
learner.learn(numTrails=18000)

np.save(WD+'para_replay.npy', network.params)
