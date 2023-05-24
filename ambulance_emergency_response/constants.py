from PIL import ImageColor

GRID_COLOR = ImageColor.getcolor('white', mode='RGB')

AGENT_COLOR = ImageColor.getcolor('blue', mode='RGB')
AMBULANCE_COLOR = ImageColor.getcolor('black', mode='RGB')

CELL_SIZE = 20

ACTION_MEANING = {
    0: "ASSIST",
    1: "NOOP",
}

REQUEST_PRIORITY = {
    0: "GREEN",
    1: "YELLOW",
    2: "RED"
}

REQUEST_WEIGHTS = [
    0.6,
    0.3,
    0.1
]

PRE_IDS = {
    'agent': 'A',
    'ambulances': 'a',
    'requests':'r',
    'wall': 'W',
    'empty': '0'
}