import logging
import random
import numpy as np
import pygame as pg

logger = logging.getLogger(__name__)

import gym
from ambulance_emergency_response.constants import *
from ambulance_emergency_response.draw import *
from gym import spaces

from ma_gym.envs.utils.action_space import MultiAgentActionSpace
from ma_gym.envs.utils.observation_space import MultiAgentObservationSpace


class AmbulanceERS(gym.Env):

    def __init__(self, grid_shape=(50, 50), n_agents=1, request_timer=100):
        
        # Infrastruture
        self.n_agents = n_agents
        self.action_space = MultiAgentActionSpace([spaces.Discrete(len(ACTION_MEANING)) for _ in range(n_agents)])
        self.observation_space = spaces.Box(low=0, high=40, shape=(5,), dtype=np.float32)

        # Display info
        self.display = None
        self.grid_shape = np.array(grid_shape)

        # Requests
        self.request_timer = request_timer
        self.request_alive = None
        
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
        for step in range(self.request_timer):
            random_num = random.randint(1, self.request_timer)

            if random_num > (step * 2) or random_num < (step ** 2):
                x = random.randint(0, self.grid_shape[0])
                y = random.randint(0, self.grid_shape[1])

                priority = random.choice(REQUEST_PRIORITY, REQUEST_WEIGHTS)
                
                request_pos.append((step, x, y, priority))
    
        features = np.array(agent_pos + request_pos).reshape(-1)

        return features

    def step(self, action):
        self.request_timer += 1
        return super().step(action)
    
    def reset(self):
        self.request_timer = 0  # ?

        initialize_render()
        self.display = pg.display.set_mode(self.grid_shape*BLOCK_SIZE)

        self.render()


    def render(self, render_mode="human"):
        
        self.display.fill(STREET_COLOR)
        draw_grid(self.display, GRID_COLOR, self.grid_shape)

        draw_agent(self.display, AGENT_COLOR, np.array([25, 25]), 18, "1")

        pg.display.update()
        

    def close(self):
        pg.quit()

