###########################################################################
#             Ambulance Emergency Response Environment Render             #
###########################################################################

import pygame, numpy
from PIL import ImageColor

from ambulance_emergency_response.settings import BLOCK_SIZE


FONT_NAME = 'Arial_bold'
FONT_SIZE = 12
FONT_COLOR = 'white'
font = None

def initialize_render():
    pygame.init()
    
    global font
    font = pygame.font.SysFont(FONT_NAME, FONT_SIZE)

def create_text(message: str, color: ImageColor):
    global font
    return font.render(message, True, color)

def draw_grid(surface: pygame.Surface, color: ImageColor, grid_size: numpy.ndarray):
    
    for x in range(0, grid_size[1] * BLOCK_SIZE, BLOCK_SIZE):
        for y in range(0, grid_size[0] * BLOCK_SIZE, BLOCK_SIZE):
            rect = pygame.Rect(x, y, BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(surface, color, rect, width=1)

def draw_agent(surface: pygame.Surface, color: ImageColor, 
               position: numpy.ndarray, radius: float, tag: str):
    
    center_position = (
        position[0]  * BLOCK_SIZE + BLOCK_SIZE / 2,
        position[1] * BLOCK_SIZE + BLOCK_SIZE / 2
    )

    pygame.draw.circle(surface, color, center_position, radius)

    surface.blit(create_text(f"A_{tag}", FONT_COLOR), center_position)

    pass

import os
import pyglet
import numpy as np
from pyglet.gl import *
import gym;

from .environment import AmbulanceERS

class CityRender(object):

    def __init__(self, grid_size=(20, 20), cell_size=1):

        self.city_size = (
            grid_size[0] * cell_size,
            grid_size[1] * cell_size
        )

        self.display_window = pyglet.window.Window(
            width=self.city_size[0], height=self.city_size[1],
            display=None, caption="Ambulance Emergency Response Challenge"
        )

        self.block_size = cell_size

        self.display_window.on_close = self.close_by_user
        self.isopen = True
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Setting resource path for pyglet
        actual_dir = os.path.dirname(__file__)
        pyglet.resource.path = [os.path.join(actual_dir, "images")]
        pyglet.resource.reindex()

        # Getting resources
        self.IMAGE_AGENGY = pyglet.resource.image("agency.png")
        self.IMAGE_AMBULANCE = pyglet.resource.image("ambulance.png")
        self.IMAGE_REQUEST = {
            0: pyglet.resource.image("patient_low_priority.png"),
            1: pyglet.resource.image("patient_middle_priority.png"),
            2: pyglet.resource.image("patient_high_priority.png"),
        }


    def close(self):
        self.display_window.close()


    def close_by_user(self):
        self.isopen = False
        exit()

    # Dynamic render for the same environment

    def __reset_render(self):
        glClearColor(*(255, 255, 255), 0)
        self.display_window.clear()
        self.display_window.switch_to()
        self.display_window.dispatch_events()
    
    def __draw_grid(self):
        batch = pyglet.graphics.Batch()

        line_references = [] # for drawing batch
        
        for offset in range(self.block_size, self.city_size[1], self.block_size):

            # column lines
            line_references.append(pyglet.shapes.Line(
                offset, 0, offset, self.city_size[0], width=2, color=(0, 230, 0), batch=batch
            ))

            # row lines
            line_references.append(pyglet.shapes.Line(
                0, offset, self.city_size[1], offset, width=2, color=(0, 230, 0), batch=batch
            ))

        batch.draw()
    
    def __draw_agencies(self, env: AmbulanceERS):
        batch = pyglet.graphics.Batch()

        sprite_references = [] # for drawing batch

        for agency in env.agencies:
            row, col = agency.position
            sprite_references.append(pyglet.sprite.Sprite(
                self.IMAGE_AGENGY,
                self.block_size * col,
                self.city_size[0] - (self.block_size * (row + 1)),
                batch=batch
            ))

        for sprite in sprite_references:
            sprite.update(scale=self.block_size / sprite.width)

        # TODO information painel containing:
        #   nr of ambulances available
        #   nr of actual assistances
        #   - perhaps metrics on a generic painel

        batch.draw()

    def __draw_requests(self, env: AmbulanceERS):
        batch = pyglet.graphics.Batch()

        sprite_references = [] # for drawing batch

        for request in env.live_requests:
            row, col = request.position
            request_priority = request.priority
            sprite_references.append(pyglet.sprite.Sprite(
                self.IMAGE_REQUEST[request_priority],
                self.block_size * col,
                self.city_size[0] - (self.block_size * (row + 1)),
                batch=batch
            ))

        for sprite in sprite_references:
            sprite.update(scale=self.block_size / sprite.width)

        batch.draw()

    def render(self, env: AmbulanceERS, return_rgb_array: bool = False):
        self.__reset_render()

        self.__draw_grid()
        self.__draw_agencies(env)  # draw agents
        self.__draw_requests(env)
        # TODO: draw ambulances

        self.display_window.flip()

        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]
            return arr

        return self.isopen
