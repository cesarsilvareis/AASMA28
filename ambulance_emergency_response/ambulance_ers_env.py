import copy
import logging
import random
from typing import Any, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

import gym
from ambulance_emergency_response.constants import *
from gym import spaces
from gym.utils import seeding

from ma_gym.envs.utils.action_space import MultiAgentActionSpace
from ma_gym.envs.utils.draw import *
from ma_gym.envs.utils.observation_space import MultiAgentObservationSpace


class AmbulanceERS(gym.Env):

    def __init__(self, grid_shape=(50, 50), request_timer=100):
        
        self.grid_shape = grid_shape
        self.action_space = spaces.Discrete(len(ACTION_MEANING))
        self.observation_space = spaces.Box(low=0, high=40, shape=(5,), dtype=np.float32)
        self.viewer = None
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
        self.request_timer = 0
    
    def render(self, render_mode="human"):
        img = draw_grid(self.grid_shape[0], self.grid_shape[1], cell_size=CELL_SIZE, fill=GRID_COLOR)
        
        # agencies
        draw_circle(image=img, pos=(1, 1), cell_size=CELL_SIZE, fill=AGENT_COLOR)
        
        draw_circle(image=img, pos=(5, 5), cell_size=CELL_SIZE, fill=AGENT_COLOR)

        # request
        draw_cell_outline(image=img, pos=(3, 3), cell_size=CELL_SIZE, fill=REQUEST_PRIORITY[0])

        img = np.asarray(img)
        if render_mode == 'rgb_array':
            return img
        elif render_mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viwer = None

