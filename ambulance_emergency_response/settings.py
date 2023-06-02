###########################################################################
#              Ambulance Emergency Response Problem SETTINGS              #
###########################################################################
from enum import Enum
from collections import OrderedDict

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
    
class IndexedOrderedDict(OrderedDict):

    def get_next_element(self, key):
        keys_list = list(self.keys())
        if key in keys_list and keys_list.index(key) < len(keys_list) - 1:
            return keys_list[keys_list.index(key) + 1]
        return None


#################################
    
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
    INVALID =4

REQUEST_IMAGE_SRC = {
    RequestPriority.HIGH:   "patient_high_priority.png",   
    RequestPriority.MEDIUM: "patient_medium_priority.png", 
    RequestPriority.LOW:    "patient_low_priority.png",
    RequestPriority.INVALID:"patient_dead.png"
}

REQUEST_WEIGHT = {
    RequestPriority.HIGH:   .1,
    RequestPriority.MEDIUM: .3,
    RequestPriority.LOW:    .6,
    RequestPriority.INVALID: 0,
}

REQUEST_DURATION_ORDER = IndexedOrderedDict({
    RequestPriority.LOW:    2,
    RequestPriority.MEDIUM: 3,
    RequestPriority.HIGH:   2,
    RequestPriority.INVALID:0,
})

# in each interval there is a 80% chance of a request happening
REQUEST_CHANCE = 80

