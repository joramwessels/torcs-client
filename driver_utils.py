# file:         driver_utils.py
# author:       Joram Wessels
# date:         11-12-2017
# dependencies: numpy
# description
# usage
#

from sys import stderr
from numpy import sign

# as stated by Torcs documentation
MAX_WHEEL_ROTATION = 21

# confusing polarities
STR_R, STR_L = -1, 1
ANG_R, ANG_L = 1, -1
DFC_R, DFC_L = -1, 1

# printing
ENABLE_DEBUG_INFO = True
PRINT_CYCLE_INTERVAL = 50 # freqency of print output in game cycles
PRINT_STATE = True
PRINT_COMMAND = False

def to_ang(ang):
    """ Steers towards the road angle
    
    Args:
        ang:    The angle of the car with the road
    
    Returns:
        The angle to steer in
    
    """
    if sign(ang) == ANG_R:
        return STR_L
    elif sign(ang) == ANG_L:
        return STR_R
    else:
        return 0
    
def away_from_ang(ang):
    """ Steers away from the road angle
    
    Args:
        ang:    The angle of the car with the road
    
    Returns:
        The angle to steer in
    
    """
    return -to_ang(ang)

def debug(iter, *args):
    """ prints debug info to stderr """
    if iter % PRINT_CYCLE_INTERVAL == 0:
        err(iter, *args)

def err(iter, *args):
    if ENABLE_DEBUG_INFO:
        spc = 6-len(str(iter))
        ovwr = 50 - len(' '.join([str(x) for x in args]))
        print(iter, ' '*spc, *args, ' '*ovwr, file=stderr)
