import pygame, numpy
from PIL import ImageColor

from ambulance_emergency_response.constants import BLOCK_SIZE


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
