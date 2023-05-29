###########################################################################
#           Ambulance Emergency Response Environment Definition           #
###########################################################################

import logging
from ambulance_emergency_response.settings import *
from collections import namedtuple, defaultdict
from enum import Enum
from itertools import product
from gym import Env, spaces
import gym
from gym.utils import seeding
from ma_gym.envs.utils.action_space import MultiAgentActionSpace
from ma_gym.envs.utils.observation_space import MultiAgentObservationSpace
import numpy as np
import random

class Entity(object):

    def __init__(self, name: str, position: tuple[int, int]):
        self.name = name
        self.position = position

class Agency(Entity):

    def __init__(self, name: str, position: tuple[int, int], num_ambulances: int):
        super().__init__(name, position)

        self.ambulances = [Ambulance(f"{PRE_IDS['ambulance']}#{name}_{i}", self) for i in range(num_ambulances)]
        self.available_ambulances = self.ambulances.copy()

        self.num_assistances_made = 0
    
    def assist(self, grid_size, goal):
        # pick up ambulance
        ambulance = self.available_ambulances.pop(0)
        ambulance.take(grid_size, goal)
        return ambulance

class Ambulance(Entity):

    def __init__(self, name: str, owner: Agency):
        super().__init__(name, owner.position)
        
        self.OWNER = owner  # should not change
        self.objective = None
        self.operating = False
        self.ongoing_path = None

    def take(self, grid_size, goal):
        self.operating = True
        self.objective = goal

        self.ongoing_path = self.__find_path_to_request(grid_size, goal)
        print(self.ongoing_path)
    
    def advance(self):
        if self.operating == False:
            return
        
        next_position = self.ongoing_path.pop(0)
        self.position = next_position
        if self.position == self.objective:
            self.operating = False
    
    def __find_path_to_request(self, grid_size, goal):
        from collections import deque
        visited = np.full(grid_size, False)
        queue = deque()
        queue.append(self.position)
        visited[self.position] = True
        parent = {}


        while queue:
            current_position = queue.popleft()
            if current_position == goal:
                path = []
                position = goal
                while (position != self.position):
                    path.append(position)
                    position = parent[position]
                path.reverse()
                return path

            neighboors_positions = self.__get_neighbor_positions(*current_position, grid_size)
            for n_pos in neighboors_positions:
                if not visited[n_pos]:
                    queue.append(n_pos)
                    visited[n_pos] = True
                    parent[n_pos] = current_position
        return None

    def __get_neighbor_positions(self, x, y, grid_size):
        return [adj_pos for adj_pos in ((x + k, y + l)
                    for k, l in ((0, -1), (-1, 0), (1, 0), (0, 1)))
                            if (0 <= adj_pos[0] < grid_size[0] and 0 <= adj_pos[1] < grid_size[1])]


class Request(Entity):

    def __init__(self, name: str, position: tuple[int, int], priority):
        super().__init__(name, position)

        self.priority = priority


class AmbulanceERS(Env):

    """
    
    """

    metadata = {"render.modes": ["human"]}


    def __init__(self, 
                 city_size : tuple[int, int] = (500, 500), 
                 num_agents: int = 1, 
                 agent_coords: list[tuple[int, int]] = [(495, 495)],
                 agent_num_ambulances: list[int] = [2],
                 request_max_generation_steps: int = 100,
                 penalty: float = 0.0,
                 sight: float = 0.5, # [0.0, 0.5]
        ):

        # TODO: input validations

        self.logger = logging.getLogger(__name__)
        self.seed()

        self.logger.info("Initializing environment...")
        
        self.current_step = 0

        self.grid_city = np.full(np.array(city_size) // BLOCK_SIZE, PRE_IDS["empty"])
        self.__log_city()

        self.agencies = []
        for i in range(num_agents):
            agency_position = tuple(np.minimum(np.array(agent_coords[i]) // BLOCK_SIZE, np.array(self.grid_city.shape) - 1))
            agent_i = Agency(f"{PRE_IDS['agent']}_" + str(i), agency_position, agent_num_ambulances[i])
            self.agencies.append(agent_i)
            self.grid_city[agent_i.position] = PRE_IDS["agent"]
        self.__log_city()

        self.penalty = penalty

        self.request_max_generation_steps = request_max_generation_steps
        
        self.request_selected = []    # a priori random selection

        available_positions = self.__get_available_positions()
        for step in range(self.request_max_generation_steps):
            if len(available_positions) == 0:
                break
            
            random_num = random.randint(1, self.request_max_generation_steps)

            if random_num < REQUEST_CHANCE:
                random_index = np.random.choice(len(available_positions))
                request_position = tuple(available_positions[random_index])

                available_positions = np.delete(available_positions, random_index, 0).tolist()

                priority = random.choices(list(REQUEST_PRIORITY.keys()), REQUEST_WEIGHTS)[0]
            
                self.request_selected.append(
                    (step, Request(f"{PRE_IDS['request']}_{step}", request_position, priority))
                )

        self.live_requests = []

        # create a copy of selected request positions to be considered further on
        self.pending_requests = [r for r in self.request_selected]

        self.active_ambulances = [] # ambulance movement

        self.rendering_initialized = False
        self.viewer = None

        self.action_space = MultiAgentActionSpace([spaces.Discrete(len(ACTION_MEANING)) for _ in range(num_agents)])
        # TODO: observation
        # self.observation_space = MultiAgentObservationSpace([self.__get_observation_space() for _ in range(num_agents)])
        self.sight = sight
        
    def __log_city(self):
        self.logger.info("[step=%d] city:\n%s", self.current_step, str(self.grid_city))
    
    def __get_available_positions(self):
        return np.argwhere(self.grid_city == PRE_IDS["empty"]).tolist()

    def __get_observation_space(self):
        pass


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self):
        self.current_step = 0

        self.render()


    def step(self, actions):

        # TODO actions (agents interactions, validations) and env logic
        for agency, action in zip(self.agencies, actions):
            match(action[0]):
                # case 0: # IDLE
                #     # self.logger.info(f"Agency: {agency.name} idle...")
                case 1: # ASSIST
                    # self.logger.info(f"Agency: {agency.name} assisting request on position: {action[1]}...")
                    ambulance = agency.assist(self.grid_city.shape, action[1])
                    self.active_ambulances.append(ambulance)
                    # ...

        #   Dynamic things
        ## Request generation
        if (self.current_step > self.request_max_generation_steps):
            # At this stage no more request will be generated
            # Agents finish the remaining requests in the city
            self.logger.info("Finishing phase...")

        if len(self.pending_requests) > 0 and self.pending_requests[0][0] == self.current_step:
            request = self.pending_requests[0][1]
            self.live_requests.append(request)

            self.pending_requests = self.pending_requests[1:]

            # update request presence in the environment
            self.grid_city[request.position] = PRE_IDS["request"]
            # self.__log_city()
            
        ## Move ambulances
        for ambulance in self.active_ambulances:
            ambulance.advance()

        self.current_step += 1


    def __init_render(self):
        from .rendering import CityRender

        self.viewer = CityRender(self.grid_city.shape, BLOCK_SIZE)
        self.rendering_initialized = True

    def render(self, mode="human"):
        if not self.rendering_initialized:
            self.__init_render()

        return self.viewer.render(self, return_rgb_array=(mode == "rgb_array"))


    def close(self):
        if self.viewer:
            self.viewer.close()
        else:
            # render not called yet
            pass


    # def simplified_features(self):

    #     current_grid = np.array(self._full_obs)

    #     agent_pos = []
    #     for agent_id in range(self.n_agents):
    #         tag = f"A{agent_id + 1}"
    #         row, col = np.where(current_grid == tag)
    #         row = row[0]
    #         col = col[0]
    #         agent_pos.append((col, row))

    #     request_pos = []
    #     for step in range(self.steps):
    #         random_num = random.randint(1, self.steps)

    #         if random_num < REQUEST_CHANCE:
    #             x = random.randint(0, self.grid_shape[0])
    #             y = random.randint(0, self.grid_shape[1])

    #             priority = random.choice(REQUEST_PRIORITY, REQUEST_WEIGHTS)
                
    #             request_pos.append((step, x, y, priority))
    
    #     features = np.array(agent_pos + request_pos).reshape(-1)

    #     return features