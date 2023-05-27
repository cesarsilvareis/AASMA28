###########################################################################
#           Ambulance Emergency Response Environment Definition           #
###########################################################################

import logging
import random
import numpy as np
import pygame as pg

logger = logging.getLogger(__name__)

import gym
from ambulance_emergency_response.settings import *
from ambulance_emergency_response.environment.rendering import *
from gym import spaces

from ma_gym.envs.utils.action_space import MultiAgentActionSpace
from ma_gym.envs.utils.observation_space import MultiAgentObservationSpace


class AmbulanceERS(gym.Env):

    def __init__(self, grid_shape=(50, 50), n_agents=1, steps=100):
        
        # Infrastruture
        self.n_agents = n_agents
        self.action_space = MultiAgentActionSpace([spaces.Discrete(len(ACTION_MEANING)) for _ in range(n_agents)])
        self.observation_space = spaces.Box(low=0, high=40, shape=(5,), dtype=np.float32)

        # Display info
        self.display = None
        self.grid_shape = np.array(grid_shape)

        # Agents positions
        self.agent_positions = []
        for _ in range(n_agents):
            while True:
                (x, y) = (random.randint(0, grid_shape[0]), random.randint(0, grid_shape[1]))
                if (x, y) not in self.agent_positions:
                    self.agent_positions.append((x, y))
                    break

        # Requests
        self.steps = steps
        self.request_timer = 0
        self.done = False
        self.request_alive = None

        self.request_pos = []
        for step in range(self.steps):
            random_num = random.randint(1, self.steps)

            if random_num < REQUEST_CHANCE:
                x = random.randint(0, self.grid_shape[0])
                y = random.randint(0, self.grid_shape[1])

                priority = random.choices(list(REQUEST_PRIORITY.keys()), REQUEST_WEIGHTS)[0]
            
                self.request_pos.append((step, x, y, priority))

        self.live_requests = []

        # create a copy
        self.pending_requests = [x for x in self.request_pos]
        
    def simplified_features(self):

        current_grid = np.array(self._full_obs)

        agent_pos = []
        for agent_id in range(self.n_agents):
            tag = f"A{agent_id + 1}"
            row, col = np.where(current_grid == tag)
            row = row[0]
            col = col[0]
            agent_pos.append((col, row))

        request_pos = []
        for step in range(self.steps):
            random_num = random.randint(1, self.steps)

            if random_num < REQUEST_CHANCE:
                x = random.randint(0, self.grid_shape[0])
                y = random.randint(0, self.grid_shape[1])

                priority = random.choice(REQUEST_PRIORITY, REQUEST_WEIGHTS)
                
                request_pos.append((step, x, y, priority))
    
        features = np.array(agent_pos + request_pos).reshape(-1)

        return features

    def step(self, actions):
        if self.request_timer > self.steps:
            self.done = True
            return
        
        if len(self.pending_requests) > 0 and self.pending_requests[0][0] == self.request_timer:
            self.live_requests.append(self.pending_requests[0])
            self.pending_requests = self.pending_requests[1:]

        if self.display:
            self.display.fill((0, 0, 0)) # clear
            self.render()

        print("Step " + str(self.request_timer))

        self.request_timer += 1

        return
    
    def reset(self):
        self.request_timer = 0  # ?

        initialize_render()
        self.display = pg.display.set_mode(self.grid_shape*BLOCK_SIZE)

        self.render()


    def render(self, render_mode="human"):
        
        self.display.fill(STREET_COLOR)
        draw_grid(self.display, GRID_COLOR, self.grid_shape)

        for position in self.agent_positions:
            draw_agent(self.display, AGENT_COLOR, np.array(position), BLOCK_SIZE/2, "1")

        for request in self.live_requests:
            draw_agent(self.display, REQUEST_COLORS[request[3]], np.array([request[1], request[2]]), BLOCK_SIZE/2, "2")

        pg.display.update()
        

    def close(self):
        pg.quit()

