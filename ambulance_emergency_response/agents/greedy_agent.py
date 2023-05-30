from aasma.agent import Agent
import numpy as np

from ambulance_emergency_response.environment.environment import Action
from ambulance_emergency_response.settings import ACTION_MEANING

class GreedyAgent(Agent):

    def __init__(self, agent_id, n_agents):
        super(GreedyAgent, self).__init__("Greedy Agent")
        self.requests_taken = []

    def action(self) -> int:
        """
        This agent will always try to assist the requests that are closer to it than the others agents.
        """
        observation = self.observation
        agencies = observation.agencies
        actions = observation.actions

        # find the self agency
        self_agency = None
        for agency in agencies:
            if agency.is_self:
                self_agency = agency
                break

        # get all requests that are closer to the self agency than the others agents
        closer_requests_actions = []
        for action in actions:
            if action.meaning == Action.assist:
                request = action.request
                
                # find the agency that is closer to the request
                closer_agency = self_agency
                for agency in agencies:
                    if agency.is_self:
                        continue
                    if self_agency.distance_to(request) < agency.distance_to(request):
                        closer_agency = agency

                # if the self agency is closer to the request than the others agents, add the action to the closer_requests list
                if closer_agency == self_agency:
                    closer_requests_actions.append(action)

        # order the closer_requests list by priority
        closer_requests_actions.sort(key=lambda request: request.priority)

        for i in range(len(closer_requests_actions)):
            if closer_requests_actions[i] not in self.requests_taken:
                return ACTION_MEANING.values().index(closer_requests_actions[i].meaning)

        return ACTION_MEANING.values().index(Action.noop)