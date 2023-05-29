from aasma.agent import Agent
import numpy as np

class GreedyAgent(Agent):

    def __init__(self, ):
        super(GreedyAgent, self).__init__("Greedy Agent")


    def action(self) -> int:
        return np.random.randint(self.n_actions)