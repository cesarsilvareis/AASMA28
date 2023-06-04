import numpy as np

from aasma.agent import Agent
from ambulance_emergency_response.environment.environment import Action, Entity
from ambulance_emergency_response.settings import ERSAction


class RandomAgent(Agent):

    def __init__(self, agency_name, n_agents):
        super(RandomAgent, self).__init__(agency_name)

    def action(self) -> int:
        """
        This agent will randomly choose an action from the available actions.
        """
        observation = self.observation

        # get random action
        actions = observation.actions
        action = np.random.choice(actions)

        return action
    