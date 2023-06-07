from aasma.agent import Agent
import numpy as np

from ambulance_emergency_response.environment.environment import Action, Entity
from ambulance_emergency_response.settings import ERSAction, REQUEST_WEIGHT

class WeakGreedyAgent(Agent):

    def __init__(self, agency_name, n_agents):
        super(WeakGreedyAgent, self).__init__(agency_name)
        self.requests_taken = []

    def action(self) -> int:
        """
        This agent will always try to assist the requests that are closer to it than the others agents.
        """
        observation = self.observation
        agencies = observation.agencies
        actions = observation.actions

        if observation.available_ambulances == 0:
            return Action(ERSAction.NOOP)

        # find the self agency
        self_agency = None
        for agency in agencies:
            if agency.name == self.name:
                self_agency = agency
                break

        # get all requests that are closer to the self agency than the others agents
        closer_requests_actions = []
        for action in actions:
            if action.meaning == ERSAction.ASSIST:
                request = action.request

                if Entity.distance_between(self_agency.position, request.position) >= request.elapse_time:
                    continue
                
                # find the agency that is closer to the request
                closer_agency = self_agency
                for agency in agencies:
                    if agency.name == self.name:
                        continue
                    if Entity.distance_between(request.position, agency.position) < Entity.distance_between(request.position, closer_agency.position):
                        closer_agency = agency

                # if the self agency is closer to the request than the others agents, add the action to the closer_requests list
                if closer_agency == self_agency:
                    closer_requests_actions.append(action)

        # order the closer_requests list by distance
        closer_requests_actions.sort(key=lambda action: Entity.distance_between(action.request.position, agency.position))

        # make sure request is not already taken
        for i in range(len(closer_requests_actions)):
            action = closer_requests_actions[i]
            if action.request not in self.requests_taken:
                self.requests_taken.append(action.request)
                return action

        return Action(ERSAction.NOOP)


class StrongGreedyAgent(Agent):

    def __init__(self, agency_name, n_agents):
        super(StrongGreedyAgent, self).__init__(agency_name)
        self.requests_taken = []

    def action(self) -> int:
        """
        This agent will always try to assist the requests that are closer to it than the others agents.
        """
        observation = self.observation
        agencies = observation.agencies
        actions = observation.actions

        if observation.available_ambulances == 0:
            return Action(ERSAction.NOOP)

        # find the self agency
        self_agency = None
        for agency in agencies:
            if agency.name == self.name:
                self_agency = agency
                break

        # get all requests that are closer to the self agency than the others agents
        closer_requests_actions = []
        for action in actions:
            if action.meaning == ERSAction.ASSIST:
                request = action.request

                if Entity.distance_between(self_agency.position, request.position) >= request.elapse_time:
                    continue
                
                # find the agency that is closer to the request
                closer_agency = self_agency
                for agency in agencies:
                    if agency.name == self.name:
                        continue
                    if Entity.distance_between(request.position, agency.position) < Entity.distance_between(request.position, closer_agency.position):
                        closer_agency = agency

                # if the self agency is closer to the request than the others agents, add the action to the closer_requests list
                if closer_agency == self_agency:
                    closer_requests_actions.append(action)

        # order the closer_requests list by priority
        closer_requests_actions.sort(key=lambda action: action.request.elapse_time)

        # make sure request is not already taken
        for i in range(len(closer_requests_actions)):
            action = closer_requests_actions[i]
            if action.request not in self.requests_taken:
                self.requests_taken.append(action.request)
                return action

        return Action(ERSAction.NOOP)
