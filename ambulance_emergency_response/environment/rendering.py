###########################################################################
#             Ambulance Emergency Response Environment RENDER             #
###########################################################################

import os

import numpy as np
import pyglet
from PIL import ImageColor
from pyglet.gl import *

from ambulance_emergency_response.settings import *

from .environment import AmbulanceERS

WORLD_PADDING = (10, 10)
METRIC_BATCH_PROPORTION = .75
METRIC_BG_COLOR = ImageColor.getcolor("#F0DFAD", mode='RGB')
METRIC_TEXT_COLOR = ImageColor.getcolor("#000000FF", mode='RGBA')
METRIC_SEP_SIZE = 50
METRIC_FONT_SIZE = 12

class CityRender(object):

    def __init__(self, grid_size=(20, 20), cell_size=1):

        self.city_size = (
            grid_size[1] * cell_size,
            grid_size[0] * cell_size
        )

        self.world_width = int((1.00 + METRIC_BATCH_PROPORTION) * self.city_size[0])
        self.world_height = self.city_size[1]

        # pivot position
        self.px, self.py = WORLD_PADDING

        self.display_window = pyglet.window.Window(
            width=self.world_width + 2 * self.px, 
            height=self.world_height + 2 * self.py,
            display=None, caption="Ambulance Emergency Response Challenge"
        )


        self.block_size = cell_size

        self.display_window.on_close = self.close_by_user
        self.isopen = True
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Setting resource path for pyglet
        actual_dir = os.path.dirname(__file__)
        pyglet.resource.path = [os.path.join(actual_dir, SRC_FILE)]
        pyglet.resource.reindex()

        # Getting resources
        self.display_window.set_icon(pyglet.resource.image(ICON_IMAGE_SRC))
        self.IMAGE_AGENGY = pyglet.resource.image(AGENCY_IMAGE_SRC)
        self.IMAGE_AMBULANCE = pyglet.resource.image(AGENCY_AMBULANCE_IMAGE_SRC)
        self.IMAGE_REQUEST = {p: pyglet.resource.image(REQUEST_IMAGE_SRC[p]) for p in REQUEST_IMAGE_SRC}


    def close(self):
        self.display_window.close()


    def close_by_user(self):
        self.isopen = False
        exit()

    # Dynamic render for the same environment

    def __reset_render(self):
        glClearColor(
            STREET_COLOR[0] / 250.0,
            STREET_COLOR[1] / 250.0,
            STREET_COLOR[2] / 250.0,
            1.0)
        self.display_window.clear()
        self.display_window.switch_to()
        self.display_window.dispatch_events()
    
    def __draw_grid(self):
        batch = pyglet.graphics.Batch()

        line_references = [] # for drawing batch
        
        # draws grid lines
        for offset in range(0, self.city_size[0] + (self.city_size[0] % self.block_size) + 1, self.block_size):

            # column lines
            line_references.append(pyglet.shapes.Line(
                self.px + offset, self.py, self.px + offset, self.py + self.city_size[1], width=2, color=GRID_COLOR, batch=batch
            ))

            # row lines
            line_references.append(pyglet.shapes.Line(
                self.px, self.py + offset, self.px + self.city_size[0], self.py + offset, width=2, color=GRID_COLOR, batch=batch
            ))

        batch.draw()
    
    def __draw_agencies(self, env: AmbulanceERS):
        batch = pyglet.graphics.Batch()

        references = [] # for drawing batch

        for i, agency in enumerate(env.agencies):
            row, col = agency.position
            sprite = pyglet.sprite.Sprite(
                self.IMAGE_AGENGY,
                self.px + self.block_size * col,
                self.py + self.city_size[0] - (self.block_size * (row + 1)),
                batch=batch
            )
            sprite.update(scale=self.block_size / sprite.width)
            sprite.color = AGENCY_COLORS[i]
            references.append(sprite)
            self.__create_label(references, batch, 
                                x=self.px + self.block_size * col,
                                y=1.5 * self.py + self.city_size[0] - (self.block_size * row),
                                text=agency.name, color=(255, 255, 255, 255), bold=True, anchor_x='center')

        batch.draw()


    def __draw_requests(self, env: AmbulanceERS):
        batch = pyglet.graphics.Batch()

        sprite_references = [] # for drawing batch

        for request in env.live_requests:
            row, col = request.position
            request_priority = request.priority
            sprite_references.append(pyglet.sprite.Sprite(
                self.IMAGE_REQUEST[request_priority],
                self.px + self.block_size * col,
                self.py + self.city_size[0] - (self.block_size * (row + 1)),
                batch=batch
            ))

        for sprite in sprite_references:
            sprite.update(scale=self.block_size / sprite.width)

        batch.draw()

    def __draw_ambulances(self, env: AmbulanceERS):
        batch = pyglet.graphics.Batch()

        sprite_references = [] # for drawing batch

        for ambulance in env.active_ambulances:
            row, col = ambulance.position
            sprite = pyglet.sprite.Sprite(
                self.IMAGE_AMBULANCE,
                self.px + self.block_size * col,
                self.py + self.city_size[0] - (self.block_size * (row + 1)),
                batch=batch
            )
            sprite.update(scale=self.block_size / sprite.width)
            sprite.color=AGENCY_COLORS[env.agencies.index(ambulance.OWNER)]
            sprite_references.append(sprite)

        batch.draw()

    def __draw_info(self, env: AmbulanceERS):
        batch = pyglet.graphics.Batch()

        x, y = self.px + self.city_size[0] + self.px, self.py

        def create_metric_label(reference: list, batch: pyglet.graphics.Batch, sep: int=0, **kwargs):
            self.__create_label(reference, batch, 
                                x=x + self.px, y=self.world_height - y - ((sep + 1) * METRIC_SEP_SIZE),
                                font_size=METRIC_FONT_SIZE, color=METRIC_TEXT_COLOR,
                                **kwargs)

        # draws background
        square = pyglet.shapes.Rectangle(x, y, self.world_width - x + self.px, 
                                self.world_height, color=METRIC_BG_COLOR, batch=batch)
        batch.draw()

        metrics_references = []

        batch = pyglet.graphics.Batch()

        metrics_references.append(pyglet.text.Label(
                text=f"Metrics:", x=x + self.px, y = self.world_height - y, 
                font_size=int(1.25 * METRIC_FONT_SIZE), bold=True, color=METRIC_TEXT_COLOR, batch=batch
        ))

        sep = 0
        for (metric, value) in env.metrics.items():
            if isinstance(value, dict): # measure by agent
                create_metric_label(metrics_references, batch, sep, text="%s:" %metric)
                sep += 1
                for agent_name in value:
                    create_metric_label(metrics_references, batch, sep, text=".%s: %.3f" %(agent_name, value[agent_name]))
                    for agent in env.agencies:
                        if agent.name == agent_name:
                            text = "NOOP" if agent.last_action.meaning == ERSAction.NOOP else "ASSIST %s" % str(agent.last_action.request.position)
                            self.__create_label(metrics_references, batch,
                                                x=x + self.px + (self.world_width - x + self.px) / 2, 
                                                y=self.world_height - y - ((sep + 1) * METRIC_SEP_SIZE),
                                                text=text, font_size=METRIC_FONT_SIZE, color=METRIC_TEXT_COLOR)
                            break
                    sep += 1
            else:
                create_metric_label(metrics_references, batch, sep, text="%s: %.3f" %(metric, value))
                sep += 1
            
        batch.draw()

    def __create_label(self, reference: list, batch: pyglet.graphics.Batch, **kwargs):
        reference.append(pyglet.text.Label(batch=batch, **kwargs))


    def render(self, env: AmbulanceERS, return_rgb_array: bool = False):
        self.__reset_render()

        self.__draw_grid()
        self.__draw_agencies(env)  # draw agents
        self.__draw_requests(env)
        self.__draw_ambulances(env)
        self.__draw_info(env)

        self.display_window.flip()

        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]
            return arr

        return self.isopen
