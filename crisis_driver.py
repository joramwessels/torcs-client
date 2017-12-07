# file:         crisis_driver.py
# author:       Joram Wessels
# date:         07-12-2017
# dependencies: -
# description
# usage
#

from pytocl.driver import Driver
from pytocl.car import State, Command

# margins of error when trying the reverse() approach
REV_CTR_MARGIN = 1
REV_ANG_MARGIN = 10
REV_ACC = 0.4
CAREFUL_REV_ACC = 0.2
MAX_WHEEL_ROTATION = 21

# NOTE
# steer left = positive
# angle left = negative
# left of the road = negative

class CrisisDriver(Driver):

    def __init__(self):
        """ Handles difficult situations in which the car gets stuck
        """
        super(CrisisDriver, self).__init__()
        self.is_in_control = False
    
    def pass_control(self, carstate):
        """ Initializes a new crisis that has not been handled yet

        Args:
            carstate:   The original carstate
        
        """
        self.is_in_control = True
        self.approach = 0
        # check object in front
        # check car in front
        # check track angle and side of the road
        # determine reverse or straight ahead

    def drive(self, carstate):
        """ Description

        Args:
            carstate:   All parameters packed in a State object
        
        Returns:
            command:    The next move packed in a Command object
        
        """
        # try different approaches and give each one a time out
        # a class level state will indicate the current approach
        # if the timer runs out, this approach will be changed
        # approach 1) reverse towards the road, then once on the road,
        #             reverse towards the wrong direction untill facing
        #             the right angle (within a certain margin)
        return command
    
    def reverse_to_middle(self, carstate, command):
        """ Reverses the car and finds it way to the middle of the road

        Args:
            carstate:       The full carstate as passed down by the server
        
        """
        command.gear = -1
        command.acceleration = REV_ACC

        #
        if is_offroad(carstate):
            angle_to_road = # TODO angle that lines back of the car facing the road
            if abs(angle_to_road) < REV_ANG_MARGIN:
                command.steering = 0
            else:
                command.steering = # TODO steer so that back of the car faces the road
                command.acceleration = CAREFUL_REV_ACC
        # if on the road
        else:
            # if in the middle of the road
            if abs(carstate.distance_from_center) < REV_CTR_MARGIN:
                if abs(carstate.angle) = < REV_ANG_MARGIN:
                    self.is_in_control = False
                    command.steering = 0
                else:
                    park(carstate, command)
            # if not in the middle of the road
            else:
                park(carstate, command)
            
        command.gear = -1
        command.acceleration = 0.25

        return command

def park(carstate, command):
    """ Parks the car backwards on the middle of the road

    Args:
        carstate:   The full car state
        command:    The command to return

    """
    if abs(carstate.angle) < MAX_WHEEL_ROTATION:
        command.steering = carstate.angle     # reverse aligns wheels with track
    else:
        angle_polarity = carstate.angle / abs(carstate.angle)
        command.steering = angle_polarity     # turns as much as possible

def car_is_stuck():
    """ Detects that the car is stuck and requires help
    
    Returns:
        A boolean indicating whether the car is stuck
    
    """
    return

def in_a_bad_place(carstate, bad_counter):
    """ Description

    Args:
        carstate:       The full carstate object
        bad_counter:    ???

    Returns:
        A boolean indicating whether the car is having troubles

    """
    something_wrong = False
    if is_offroad(carstate):
        #print("I'm offroad!")
        something_wrong = True
    if is_reversed(carstate):
        #print("I'm reversed!")
        something_wrong = True
    if is_blocked(carstate):
        #print("I'm blocked!")
        something_wrong = True
    if (something_wrong):
        bad_counter += 1
    else:
        bad_counter = 0
    # if we have been in a bad place for 2 seconds
    if bad_counter >= 100:
        return True
    return False

def is_offroad(carstate):
    """ Returns True if car is off road """
    return max(carstate.distances_from_edge) == -1

def is_blocked(carstate):
    """ Returns True if car is stuck """
    return abs(carstate.speed_x) <= 5 and carstate.current_lap_time >= 10

def is_reversed(carstate):
    """ Returns True if the car is facing the wrong side """
    return abs(carstate.angle) >= 90

def err(*args):
    """ prints to standard error """
    print(*args, file=stderr)