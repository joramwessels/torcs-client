# file:         crisis_driver.py
# author:       Joram Wessels
# date:         07-12-2017
# dependencies: numpy, pytocl
# description
# usage
#

from pytocl.driver import Driver
from pytocl.car import State, Command

from sys import stderr
from numpy import sign

# margins of error when trying the reverse_to_middle approach
CTR_ANG_MARGIN = 10
PRP_ANG_MARGIN = 10

# careful acceleration values during crisis handling
ACC              = 0.7
REV_ACC          = 0.5
OFF_ROAD_ACC     = 0.2
OFF_ROAD_REV_ACC = 0.3

# as stated by Torcs documentation
MAX_WHEEL_ROTATION = 21

# timers (100 = 2 sec)
BAD_COUNTER_THRESHOLD = 100
BAD_COUNTER_MANUAL_THRESHOLD = 600
APPROACH_TIMEOUT = 1000

# NOTE
# steer left = positive
# angle left = negative
# left of the road = negative

class CrisisDriver(Driver):

    def __init__(self, logdata=True):
        """ Handles difficult situations in which the car gets stuck
        """
        super(CrisisDriver, self).__init__(logdata=logdata)
        self.iter = 0
        self.driver = Driver(logdata=False)
        self.is_in_control = False
        self.approaches = [self.navigate_to_middle, self.original_implementation]
        self.bad_counter = 0
        self.needs_help = False

    def drive(self, carstate):
        """ Gets the car out of a difficult situation

        Tries different approaches and gives each one a time out.
        A class level variable keeps track of the current approach
        across game cycles. If the timer runs out, the approach
        is terminated and the next approach gets initiated.

        Args:
            carstate:   All parameters packed in a State object
        
        Returns:
            command:    The next move packed in a Command object
        
        """
        command = Command()
        if self.appr_counter < APPROACH_TIMEOUT:
            self.next_approach()
        return self.approaches[self.approach](carstate, command)
    
    def pass_control(self, carstate):
        """ Initializes a new crisis that has not been handled yet

        Args:
            carstate:   The original carstate
        
        """
        err(self.iter,"CRISIS: control received")
        self.is_in_control = True
        self.approach = 0
        self.appr_counter = 0
        self.previous_angles = []
        # check if car in front
        # check if car behind
        # check track angle and side of the road
        # determine reverse or straight ahead
    
    def return_control(self):
        """ Passes control back to the main driver """
        err(self.iter,"CRISIS: control returned")
        self.is_in_control = False
    
    def next_approach(self):
        """ Adjusts state to next approach """
        err(self.iter,"CRISIS: next approach:",
            self.approaches[self.approach].__name__)
        self.approach += 1
        self.appr_counter = 0
        if self.approach >= len(self.approaches):
            self.return_control()
    
    def approach_succesful(self):
        """ Called when a technique finished executing
        """
        err(self.iter,"CRISIS: approach succesful:",
            self.approaches[self.approach].__name__)
        if self.has_problem:
            self.next_approach()
        else:
            self.return_control()
    
    def update_status(self, carstate):
        """ Updates the status of the car regarding its problems

        Args:
            carstate:   The full carstate
        
        """
        self.iter += 1
        self.previous_angles.append(carstate.angle)

        self.is_on_left_side  = sign(carstate.distance_from_center) == -1
        self.is_on_right_side = not self.is_on_left_side
        self.faces_left   = sign(carstate.angle) == -1
        self.faces_right  = not self.faces_left
        self.faces_front  = abs(carstate.angle) < 90
        self.faces_back   = not self.faces_front
        self.faces_middle = self.is_on_left_side == self.faces_right

        self.is_standing_still = carstate.speed_x == 0
        self.has_car_in_front  = car_in_front(carstate.opponents)
        self.has_car_behind    = car_behind(carstate.opponents)
        self.is_blocked        = self.is_standing_still and \
                          (self.has_car_in_front or self.has_car_behind) # TODO when accelerating but speed < 1?
        self.is_off_road = sign(max(carstate.distances_from_edge)) == -1 # TODO verify
        self.is_reversed = self.faces_back
        self.is_going_in_circles = going_in_circles(self.previous_angles)
        self.has_problem = self.is_off_road or \
                           self.is_going_in_circles or \
                           self.is_blocked
        if self.has_problem:
            self.bad_counter += 1
            if self.bad_counter >= BAD_COUNTER_THRESHOLD:
                self.needs_help = True
        else:
            self.bad_counter = 0
            self.needs_help = False

    #
    # Approach Implementations
    #
    def original_implementation(self, carstate, command):
        """

        approach 0)

        Args:
            carstate:       The full carstate as passed down by the server
            command:        The command to adjust
        
        """
        command = self.driver.drive(carstate)
        is_stuck = abs(carstate.speed_x) <= 5 and carstate.current_lap_time >= 10
        if self.bad_counter >= BAD_COUNTER_MANUAL_THRESHOLD and is_stuck:
            # we try reversing
            command.gear = -command.gear
            if command.gear < 0:
                command.steering = -command.steering
                command.gear = -1
            self.bad_counter = 200
        return command
    
    def navigate_to_middle(self, carstate, command):
        """ Finds it way to the middle of the road by driving in reverse
        
        approach 1) reverse towards the road, then once on the road,
                    reverse towards the the middle until facing foward
                    with an angle that's within the margin

        Args:
            carstate:       The full carstate as passed down by the server
            command:        The command to adjust
        
        """
        if self.is_off_road:
            perp_angle = 90 * -sign(carstate.distance_from_center)
            if self.faces_middle:
                diff_with_perp_angle = perp_angle - carstate.angle
                if not abs(diff_with_perp_angle) < PRP_ANG_MARGIN:
                    command.steering = -sign(diff_with_perp_angle)
                gear = 1
                acceleration = OFF_ROAD_ACC
            else:
                diff_with_perp_angle = perp_angle + carstate.angle
                if not abs(diff_with_perp_angle) < PRP_ANG_MARGIN:
                    command.steering = -sign(diff_with_perp_angle)
                gear = -1
                acceleration = OFF_ROAD_REV_ACC
        else:
            if abs(carstate.angle) < CTR_ANG_MARGIN:
                self.approach_succesful()
            else:
                command.steering = -sign(carstate.distance_from_center)
                if self.faces_middle:
                    command.gear = 1
                    command.acceleration = ACC
                else:
                    command.gear = -1
                    command.acceleration = REV_ACC
        return command

#
# Problem Detectors
#
def car_in_front(opp):
    """ Returns True if there's a car in front of ours """
    return any([o < 1 for o in opp[17:18]])        # TODO are these the 20deg slices in front?

def car_behind(opp):
    """ Returns True if there's a car behind ours """
    return any([0 < 0 for o in opp[0] + opp[35]])  # TODO and these in the back?

def going_in_circles(angles):
    """ Returns True if the car is rapidly going round in circles """
    # TODO checks if there is a pattern in the angles that
    # constitutes rotation. Take care of skip from 180 to -180.
    return False




def err(*args):
    """ prints to standard error """
    print(*args, file=stderr)