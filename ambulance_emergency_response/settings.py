from PIL import ImageColor

STREET_COLOR = ImageColor.getcolor('#2C4251', mode='RGB')
GRID_COLOR = ImageColor.getcolor("#C1C1C1", mode='RGB')

BLOCK_SIZE = 35  # square

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

# in each interval there is a 80% chance of a request happening
REQUEST_CHANCE = 80

PRE_IDS = {
    'agent': 'A',
    'ambulance': 'a',
    'request':'r',
    'wall': 'W',
    'empty': '0'
}