###########################################################################
#              Ambulance Emergency Response Problem SETTINGS              #
###########################################################################
from enum import Enum

from PIL import ImageColor


class ExtendedEnum(Enum):

    def __str__(self):
        return str(self.name)

    @classmethod
    def count(cls: Enum) -> int:
        return len(cls)

    @classmethod
    def list(cls: Enum) -> list:
        return list(map(lambda c: c.value, cls))
    
SRC_FILE = "images/"

#################################
#           WORLD/ENV           #
#################################

STREET_COLOR = ImageColor.getcolor('#2C4251', mode='RGB')
GRID_COLOR = ImageColor.getcolor("#C1C1C1", mode='RGB')

BLOCK_SIZE = 35  # square

class ERSEntity(ExtendedEnum):
    AGENCY      =1
    AMBULANCE   =2
    REQUEST     =3
    NONE        =4

ENTITY_IDS = {
    ERSEntity.AGENCY:   'A',
    ERSEntity.AMBULANCE:'a',
    ERSEntity.REQUEST:  'r',
    ERSEntity.NONE:     '0',
}

class ERSAction(ExtendedEnum):
    ASSIST  =1
    NOOP    =2


#################################
#            Agencies           #
#################################
AGENCY_IMAGE_SRC =              "agency.png"
AGENCY_AMBULANCE_IMAGE_SRC =    "ambulance.png"


#################################
#           Requests            #
#################################
class RequestPriority(ExtendedEnum):
    HIGH    =1
    MEDIUM  =2
    LOW     =3

REQUEST_IMAGE_SRC = {
    RequestPriority.HIGH:   "patient_high_priority.png",   
    RequestPriority.MEDIUM: "patient_medium_priority.png", 
    RequestPriority.LOW:    "patient_low_priority.png",
}

REQUEST_WEIGHT = {
    RequestPriority.HIGH:   .1,
    RequestPriority.MEDIUM: .3,
    RequestPriority.LOW:    .6,
}

REQUEST_DURATION_ORDER = [
    RequestPriority.LOW,
    RequestPriority.MEDIUM,
    RequestPriority.HIGH
]

# in each interval there is a 80% chance of a request happening
REQUEST_CHANCE = 80

