###########################################################################
#           Ambulance Emergency Response Environment DEFINITION           #
###########################################################################

import logging
import random
from collections import namedtuple

import numpy as np
from gym import Env, spaces
from gym.utils import seeding
from ma_gym.envs.utils.action_space import MultiAgentActionSpace
from ma_gym.envs.utils.observation_space import MultiAgentObservationSpace

from ambulance_emergency_response.settings import *


class Message(object):

        def __init__(self, request: 'Request', sender: str):
            self.request = request
            self.sender = sender
    
        def __str__(self):
            return f"{self.sender.name} -> {self.request}"
    
        def __repr__(self):
            return f"{self.sender.name} -> {self.request}"

class Action():

    def __init__(self,  meaning: ERSAction, request: 'Request'=None):
        self.request = request
        self.meaning = meaning
        self.message = None

    def attach_message(self, message: Message):
        self.message = message
        return self

    def __str__(self):
        return f"{self.meaning} {self.request}" if self.request else f"{self.meaning}"
    
    def __repr__(self):
        return str(self)

    # @staticmethod
    # def get_str_int(str: str) -> int:
    #     return [key for key, value in ACTION_MEANING.items() if value == str][0]

class Entity(object):

    GRID_SIZE : tuple[int, int]

    def __init__(self, name: str, position: tuple[int, int]):
        self.name = name
        self.position = position

    def distance_to(self, other: 'Entity'):
        return np.linalg.norm(np.array(self.position) - np.array(other.position))
    
    @staticmethod
    def distance_between(position1: tuple[int, int], position2: tuple[int, int]):
        return np.linalg.norm(np.array(position1) - np.array(position2))
    
    @staticmethod
    def in_world(position: tuple[int, int]):
        return 0 <= position[0] < Entity.GRID_SIZE[0] and 0 <= position[1] < Entity.GRID_SIZE[1]

class Agency(Entity):

    def __init__(self, name: str, position: tuple[int, int], num_ambulances: int):
        super().__init__(name, position)

        self.ambulances = [Ambulance(f"{ENTITY_IDS[ERSEntity.AMBULANCE]}#{name}_{i}", self) for i in range(num_ambulances)]
        self.available_ambulances = self.ambulances.copy()
        self.reward = 0

        self.num_assistances_made = 0
    
    def assist(self, request: 'Request') -> 'Ambulance':
        # pick up ambulance
        if len(self.available_ambulances) == 0:
            return None
        ambulance = self.available_ambulances.pop(0)
        ambulance.take(request.position, request)
        return ambulance
    
    def retrieve_ambulance(self, ambulance: 'Ambulance'):
        self.available_ambulances.append(ambulance)

    def reset(self):
        self.num_assistances_made = 0
        self.available_ambulances = self.ambulances.copy()
        for ambulance in self.ambulances:
            ambulance.reset()

class Ambulance(Entity):

    def __init__(self, name: str, owner: Agency):
        super().__init__(name, owner.position)
        
        self.OWNER = owner  # should not change
        self.objective = None
        self.operating = False
        self.ongoing_path = None
        self.coming_back = False
        self.request : 'Request' = None

    def take(self, goal, request: 'Request'=None):

        if request is not None:
            self.request = request

        self.ongoing_path = self.__find_path_to_request(goal)
        if self.ongoing_path is not None:
            self.operating = True
            self.objective = goal
    
    def advance(self) -> bool:
        if len(self.ongoing_path) == 0:
            return False
        if not self.operating:
            return False
        
        if self.request.priority == RequestPriority.INVALID and not self.coming_back:
            if self.position == self.OWNER.position:
                self.operating = False
                return True
            self.coming_back = True
            self.take(self.OWNER.position)
        
        if len(self.ongoing_path) == 0:
            print("DEBUG:", self.position, self.OWNER.position, self.request.position, self.objective)
        next_position = self.ongoing_path.pop(0)
        self.position = next_position
        if self.position == self.objective:
            if self.position == self.OWNER.position:
                self.operating = False
                self.coming_back = False
                return True
            else:
                self.take(self.OWNER.position)
                self.coming_back = True
                return True
        return False
    
    def __find_path_to_request(self, goal):
        from collections import deque
        visited = np.full(Entity.GRID_SIZE, False)
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

            neighboors_positions = self.__get_neighbor_positions(*current_position)
            for n_pos in neighboors_positions:
                if not visited[n_pos]:
                    queue.append(n_pos)
                    visited[n_pos] = True
                    parent[n_pos] = current_position
        return None

    def __get_neighbor_positions(self, x, y):
        return [adj_pos for adj_pos in ((x + k, y + l)
                    for k, l in ((0, -1), (-1, 0), (1, 0), (0, 1)))
                            if (Entity.in_world(adj_pos))]

    def reset(self):
        self.objective = None
        self.operating = False
        self.ongoing_path = None
        self.request = None
        self.coming_back = False
        self.position = self.OWNER.position


class Request(Entity):

    def __init__(self, name: str, position: tuple[int, int], priority: RequestPriority):
        super().__init__(name, position)

        self.priority = priority
        self.time_alive = REQUEST_DURATION_ORDER[priority]
        self.elapse_time = 0
        priority_copy = priority
        while (priority_copy != RequestPriority.INVALID):
            self.elapse_time += REQUEST_DURATION_ORDER[priority_copy]
            priority_copy = REQUEST_DURATION_ORDER.get_next_element(priority_copy)

    def time_step(self):
        if self.priority == RequestPriority.INVALID:
            return
        
        self.time_alive -= 1
        self.elapse_time -= 1

        if self.time_alive == 0:
            self.priority = REQUEST_DURATION_ORDER.get_next_element(self.priority)
            self.time_alive = REQUEST_DURATION_ORDER[self.priority]


    def __str__(self) -> str:
        return f"{self.name} {self.position} {self.priority}"
    
    def __repr__(self) -> str:
        return f"{self.name} {self.position} {self.priority}"

    def __eq__(self, other: 'Request'):
        return self.name == other.name and self.priority == other.priority


class AmbulanceERS(Env):

    """
    
    """

    metadata = {"render.modes": ["human"]}

    Observation = namedtuple(
        "Observation",
        ["field", "actions", "agencies", "available_ambulances", "total_ambulances", "current_step", "messages",],
    )
    AgentObservation = namedtuple(
        "PlayerObservation", 
        ["is_self", "position", "reward"]
    )

    def __init__(self, 
                 city_size : tuple[int, int] = (500, 500), 
                 num_agents: int = 1, 
                 agent_coords: list[tuple[int, int]] = [(495, 495)],
                 agent_num_ambulances: list[int] = [2],
                 occupancy_map: list[list[int]] | None=None,
                 request_max_generation_steps: int = 100,
                 penalty: float = 0.0,
                 sight: float = 1.0, # [0.0, 1.0]
                 show_density_map: bool=False
        ):

        self.N_AGENTS = num_agents

        # TODO: input validations

        self.logger = logging.getLogger(__name__)
        self.seed()

        self.logger.info("Initializing environment...")
        
        self.current_step = 0

        self.grid_city = np.full(np.array(city_size) // BLOCK_SIZE, ENTITY_IDS[ERSEntity.NONE])
        Entity.GRID_SIZE = self.grid_city.shape

        if occupancy_map is None:
            self.OCCUPANCY_MAP = np.random.uniform(0, 1 / (self.grid_city.shape[0] * self.grid_city.shape[1]), size=self.grid_city.shape)
        else:
            if occupancy_map.shape != self.grid_city.shape:
                raise ValueError("Invalid occupancy map layout size. Must be %", self.grid_city.shape)
            self.OCCUPANCY_MAP = occupancy_map
        
        if show_density_map:
            self.__plot_occupancy_map()
        

        self.agencies : list[Agency] = []
        for i in range(self.N_AGENTS):
            agency_position = tuple(np.minimum(np.array(agent_coords[i]) // BLOCK_SIZE, np.array(self.grid_city.shape) - 1))
            agent_i = Agency(f"{ENTITY_IDS[ERSEntity.AGENCY]}_" + str(i), agency_position, agent_num_ambulances[i])
            self.agencies.append(agent_i)
            self.grid_city[agent_i.position] = ENTITY_IDS[ERSEntity.AGENCY]

        self.penalty = penalty

        self.request_max_generation_steps = request_max_generation_steps
        
        self.live_requests : list[Request] = []
        self.num_spawned_requests = 0

        self.active_ambulances : list[Ambulance] = [] # ambulance movement
        self.finishing_phase = False
        self.end = False

        self.rendering_initialized = False
        self.viewer = None

        self.sight = sight
        self.action_space = MultiAgentActionSpace([spaces.Discrete(ERSAction.count()) for _ in range(num_agents)])
        self.observation_space = MultiAgentObservationSpace([self.__get_observation_space() for _ in range(num_agents)])

        ## Metrics
        self.num_taken_requests = 0
        self.finalized_requests = 0
        self.total_time_alive_requests = 0
        self.total_number_ambulances = sum([len(agency.ambulances) for agency in self.agencies])

        self.metrics = {
            "Response-rate": 0.00,
            "Response-time": 0.00,
            "Ambulance-availability": float(self.total_number_ambulances),
            "Resource-utilization": { agency.name: 0.00 for agency in self.agencies }
        }

        ## Message passing
        self.message_queue = []


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
       
    def __plot_occupancy_map(self):
        from matplotlib import pyplot as plt, cm, colors
        plt.figure(figsize=tuple(np.array(self.OCCUPANCY_MAP.shape) // 2))

        color_list = ['green', 'yellow', 'red']
        cmap = colors.LinearSegmentedColormap.from_list('custom_cmap', colors=color_list)

        plt.title('Gradient map')
        plt.imshow(self.OCCUPANCY_MAP, origin='upper', interpolation="none", cmap=cmap)
        plt.xticks(np.arange(self.OCCUPANCY_MAP.shape[0]), np.arange(self.OCCUPANCY_MAP.shape[0]))  # need to set the ticks manually
        plt.yticks(np.arange(self.OCCUPANCY_MAP.shape[1]), np.arange(self.OCCUPANCY_MAP.shape[1] - 1, -1, -1))
        plt.colorbar(label="Request spawn frequency")
        plt.show()


    def __log_city(self):
        self.logger.info("[step=%d] city:\n%s", self.current_step, str(self.grid_city))
    
    def __get_available_positions(self) -> list[tuple[int, int]]:
        position_list = np.argwhere(self.grid_city == ENTITY_IDS[ERSEntity.NONE]).tolist()
        return list(map(tuple, position_list))

    def __get_closest_agency(self, request: Request) -> Agency:
        return min(self.agencies, key=lambda agency: np.linalg.norm(np.array(agency.position) - np.array(request.position)))

    def __get_observation_space(self):
        grid_shape = np.array(np.array(self.grid_city.shape) * self.sight, dtype=int)

        # whole grid positions (request and agent positions)
        grid_obs_low = np.zeros(grid_shape, dtype=np.float32)
        grid_obs_high = np.ones(grid_shape, dtype=np.float32)

        return spaces.Box(grid_obs_low, grid_obs_high)

    def __get_valid_actions(self, agent: Agency):
        actions = [Action(ERSAction.NOOP)]

        for request in self.live_requests:
            if request.priority == RequestPriority.INVALID:
                continue
            # check if this agency is the closest to the request
            if self.__get_closest_agency(request) == agent:
                actions.append(Action(ERSAction.ASSIST, request))

        return actions
    
    def __get_messages(self, agent: Agency):
        return [message for message in self.message_queue if message.sender != agent]

    def __make_obs(self, agent: Agency):
        return self.Observation(
            field=np.copy(self.grid_city),
            actions=self.__get_valid_actions(agent),
            agencies=[agency for agency in self.agencies],
            available_ambulances=len(agent.available_ambulances),
            total_ambulances=len(agent.ambulances),
            current_step=self.current_step,
            messages=self.__get_messages(agent),
        )
    
    def __make_gym_obs(self):
        def get_agency_reward(observation):
            for agency in observation.agencies:
                if agency.is_self:
                    return agency.reward
        
        nobs = { agency.name: self.__make_obs(agency) for agency in self.agencies}
        # nreward = [get_agency_reward(observation=o) for o in nobs.values()]
        terminal = self.end
        ninfo = [{"observation": o} for o in nobs]
        
        return nobs, [], terminal, ninfo

    def reset(self):
        self.current_step = 0
        self.live_requests.clear()
        self.num_spawned_requests = 0
        self.finishing_phase = False
        self.end = False

        for agency in self.agencies:
            agency.reset() # this also resets ambulances
        
        self.active_ambulances.clear()

        self.num_taken_requests = 0
        self.finalized_requests = 0
        self.total_time_alive_requests = 0
        self.total_number_ambulances = sum([len(agency.ambulances) for agency in self.agencies])

        self.metrics = {
            "Response-rate": 1.00,
            "Response-time": 0.00,
            "Ambulance-availability": float(self.total_number_ambulances),
            "Resource-utilization": { agency.name: 0.00 for agency in self.agencies }
        }
        
        return { agency.name: self.__make_obs(agency) for agency in self.agencies}


    def step(self, actions):

        self.message_queue.clear()

        for agency, action in zip(self.agencies, actions):
            if action.message is not None:
                self.message_queue.append(action.message)
            match(action.meaning):
                case ERSAction.NOOP:
                    self.logger.info(f"Agency: {agency.name} idle...")
                case ERSAction.ASSIST:
                    request = action.request
                    self.logger.info(f"Agency: {agency.name} assisting request on position: {request.position}...")
                    ambulance = agency.assist(request)
                    if ambulance is not None:
                        self.active_ambulances.append(ambulance)
                    # ...

        ##   Dynamic things

        # Request generation
        if not self.finishing_phase:
            if self.num_spawned_requests >= self.request_max_generation_steps:
                self.finishing_phase = True
                self.logger.info("Finishing phase...")
            
            else:
                for valid_position in self.__get_available_positions():
                    if random.random() >= self.OCCUPANCY_MAP[valid_position[0]][valid_position[1]] * 2:
                        continue

                    # generate new request
                    request = Request(
                        f"{ENTITY_IDS[ERSEntity.REQUEST]}_{self.current_step}_{valid_position}",
                        valid_position,
                        random.choices(*zip(*REQUEST_WEIGHT.items()))[0]
                    )
                    self.live_requests.append(request)
                    self.num_spawned_requests += 1
                    self.grid_city[request.position] = ENTITY_IDS[ERSEntity.REQUEST]
            
        # Checks invalid requests firstly
        for request in self.live_requests:
            if request.priority == RequestPriority.INVALID:
                self.live_requests.remove(request)
                self.finalized_requests += 1
                self.grid_city[request.position] = ENTITY_IDS[ERSEntity.NONE]

        # Move ambulances
        for ambulance in self.active_ambulances:
            if (self.grid_city[ambulance.position] == ENTITY_IDS[ERSEntity.AMBULANCE]):
                self.grid_city[ambulance.position] = ENTITY_IDS[ERSEntity.NONE]
            reached_goal = ambulance.advance()
            self.grid_city[ambulance.position] = ENTITY_IDS[ERSEntity.AMBULANCE]

            # if reached patient request
            if reached_goal and ambulance.operating and ambulance.request in self.live_requests:
                self.total_time_alive_requests += ambulance.request.time_alive
                self.num_taken_requests += 1
                ambulance.OWNER.num_assistances_made += 1
                self.live_requests.remove(ambulance.request)
                self.finalized_requests += 1
                
            # if reached owner agency
            elif reached_goal and not ambulance.operating:
                self.active_ambulances.remove(ambulance)
                ambulance.OWNER.retrieve_ambulance(ambulance)
                self.grid_city[ambulance.position] = ENTITY_IDS[ERSEntity.AGENCY]

        # Update requests' timer
        for request in self.live_requests:
            request.time_step()

        for agency in self.agencies:
            self.total_number_ambulances += len(agency.available_ambulances)

        # TODO: Update metrics
        if (self.finalized_requests):
            self.metrics["Response-rate"] = self.num_taken_requests / self.finalized_requests

        if (self.num_taken_requests):
            self.metrics["Response-time"] = self.total_time_alive_requests / self.num_taken_requests
            for agency in self.agencies:
                self.metrics["Resource-utilization"][agency.name] = agency.num_assistances_made / self.num_taken_requests

        self.metrics["Ambulance-availability"] = self.total_number_ambulances / (self.current_step + 2)


        # self.__log_city()
        self.current_step += 1

        if self.finishing_phase and self.live_requests == self.active_ambulances == []:
            self.end = True

        # TODO: change second argument (reward)
        return self.__make_gym_obs()


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
