from aasma.agent import Agent
import numpy as np

from ambulance_emergency_response.environment.environment import Action, Entity, Agency, Request, Message
from ambulance_emergency_response.settings import ERSAction, REQUEST_WEIGHT

class ConventionAgent(Agent):

    # social convention is a list of angeny names
    def __init__(self, agency_name, n_agents, social_convention):
        super(ConventionAgent, self).__init__(agency_name)
        self.social_convention = social_convention
        self.requests_taken = []
        self.requests_sent = []

    def action(self) -> int:
        """
        This agent will always try to assist the requests that were sent by other agents, according to the social convention.
        """
        observation = self.observation
        agencies = observation.agencies
        actions = observation.actions
        messages = observation.messages
        agent_order = self.social_convention

        message = None

        # find agency name next to own name in social convention
        self_index = agent_order.index(self.name)
        next_index = (self_index + 1) % len(agent_order)
        next_agency_name = agent_order[next_index]

        # find the self agency
        self_agency = None
        for agency in agencies:
            if agency.is_self:
                self_agency = agency
                break

        # get all requests sent by the next agency
        next_agency_requests = []
        for message in messages:
            if Entity.distance_between(self_agency.position, message.request.position) >= message.request.elapse_time:
                continue
            if message.sender == next_agency_name and message.request not in self.requests_taken:
                next_agency_requests.append(message.request)

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
                    if agency.is_self:
                        continue
                    if Entity.distance_between(request.position, agency.position) < Entity.distance_between(request.position, closer_agency.position):
                        closer_agency = agency

                # if the self agency is closer to the request than the others agents, add the action to the closer_requests list
                if closer_agency == self_agency and request not in self.requests_taken:
                    closer_requests_actions.append(action)

        # order the requests list by priority
        closer_requests_actions.sort(key=lambda action: REQUEST_WEIGHT[action.request.priority], reverse=True)
        next_agency_requests.sort(key=lambda request: REQUEST_WEIGHT[request.priority], reverse=True)

        if observation.available_ambulances < observation.total_ambulances / 2 and len(closer_requests_actions) > 0:
            message = Message(closer_requests_actions[0].request, self.name)
            self.requests_sent.append(closer_requests_actions[0].request)

        if observation.available_ambulances == 0:
            return Action(ERSAction.NOOP).attach_message(message)

        if len(next_agency_requests) > 0:
            request = next_agency_requests[0]
            self.requests_taken.append(request)
            return Action(ERSAction.ASSIST, request).attach_message(message)

        for i in range(len(closer_requests_actions)):
            action = closer_requests_actions[i]
            if action.request not in self.requests_sent:
                self.requests_taken.append(action.request)
                return action.attach_message(message)

        return Action(ERSAction.NOOP).attach_message(message)
