from aasma.agent import Agent
import numpy as np

from ambulance_emergency_response.environment.environment import Action, Entity
from ambulance_emergency_response.settings import ACTION_MEANING

class RandomAgent(Agent):

    def __init__(self, agency_name, n_agents):
        super(RandomAgent, self).__init__(agency_name)
        self.requests_taken = []

    def action(self) -> int:
        """
        This agent will randomly choose an action from the available actions.
        """
        observation = self.observation
        
        if observation.available_ambulances == 0:
            return Action(Action.noop)

        # get random action
        actions = observation.actions
        action = np.random.choice(actions)

        if action.meaning == Action.assist:
            request = action.request
            self.requests_taken.append(request)

        return action
    