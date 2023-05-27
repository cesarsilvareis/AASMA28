from PIL import ImageColor

STREET_COLOR = ImageColor.getcolor('white', mode='RGB')
GRID_COLOR = ImageColor.getcolor('black', mode='RGB')

AGENT_COLOR = ImageColor.getcolor('orange', mode='RGB')
AMBULANCE_COLOR = ImageColor.getcolor('black', mode='RGB')

BLOCK_SIZE = 20  # square

ACTION_MEANING = {
    0: "ASSIST",
    1: "NOOP",
}

REQUEST_PRIORITY = {
    0: "GREEN",
    1: "YELLOW",
    2: "RED"
}

REQUEST_COLORS = {
    0: ImageColor.getcolor('green', mode='RGB'),
    1: ImageColor.getcolor('yellow', mode='RGB'),
    2: ImageColor.getcolor('red', mode='RGB')
}

REQUEST_WEIGHTS = [
    0.6,
    0.3,
    0.1
]

# in each interval there is a 80% chance of a request happening
REQUEST_CHANCE = 80

PRE_IDS = {
    'agent': 'A',
    'ambulances': 'a',
    'requests':'r',
    'wall': 'W',
    'empty': '0'
}