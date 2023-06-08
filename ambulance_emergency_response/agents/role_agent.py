from aasma.agent import Agent
import numpy as np

from ambulance_emergency_response.environment.environment import Action, Entity, Agency, Request, Message
from ambulance_emergency_response.settings import ERSAction, REQUEST_WEIGHT

class RoleAgent(Agent):

    def __init__(self, agency_name, n_agents):
        super(RoleAgent, self).__init__(agency_name)
        self.requests_taken = []
        self.requests_sent = []

    def action(self) -> int:
        """
        This agent will always try to assist the requests that were sent by other agents, according to the role that is assigned to him.
        """
        observation = self.observation
        agencies = observation.agencies
        actions = observation.actions
        messages = observation.messages

        message = None

        # find the self agency
        self_agency = None
        for agency in agencies:
            if agency.name == self.name:
                self_agency = agency
                break

        # get all requests sent by the next agency
        next_agency_requests = []
        for message in messages:
            if Entity.distance_between(self_agency.position, message.request.position) >= message.request.elapse_time:
                continue
            if message.request not in self.requests_taken and self.__has_better_potential(observation, message):
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
                    if agency.name == self.name:
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
            message = Message(closer_requests_actions[-1].request, self.name)
            self.requests_sent.append(closer_requests_actions[-1].request)

        # checks for GRAB actions first prioritizing closer ambulances
        best_grab_action = None
        for action in actions:
            if action.meaning == ERSAction.GRAB:
                ambulance = action.ambulance
                if ambulance.coming_back:
                    continue

                if not best_grab_action:
                    best_grab_action = action
                    continue

                ambulance = action.ambulance
                if Entity.distance_between(self_agency.position, ambulance.position) < \
                    Entity.distance_between(self_agency.position, best_grab_action.ambulance.position):
                    best_grab_action = action
        
        if best_grab_action:
            return best_grab_action

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
    


    def __has_better_potential(self, observation, message: Message):
        request = message.request

        self_potential = None
        for agency in observation.agencies:
            if agency.name == self.name:
                self_potential = self.__potencial(agency, request)
                break

        for agency in observation.agencies:
            if agency.name == self.name or agency.name == message.sender:
                continue
            if self_potential < self.__potencial(agency, request):
                return False
            if self_potential == self.__potencial(agency, request) and self.name < agency.name:
                return False
            
        return True
    
    @staticmethod
    def __potencial(agency: Agency, request: Request):
        return (1 / Entity.distance_between(agency.position, request.position)) # FIXME