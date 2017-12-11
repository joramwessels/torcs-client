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

from driver_utils import *

# margins of error when trying the navigate_to_middle approach
CTR_ANG_MARGIN = 10
PRP_ANG_MARGIN = 10

# careful acceleration values during crisis handling
ACC              = 0.1
REV_ACC          = 0.1
OFF_ROAD_ACC     = 0.1
OFF_ROAD_REV_ACC = 0.2

# timers (100 = 2 sec)
BAD_COUNTER_THRESHOLD = 100
BAD_COUNTER_MANUAL_THRESHOLD = 600
APPROACH_TIMEOUT = 1000

# cross-cycle trackers
MAX_ANGLES = 64 # amount of angles to keep track of
MAX_ACCELS = 16  # amount of cycles in which you should have moved
MAX_GEARS = MAX_ACCELS

# TODO
# - back off when blocked
# NOTE
# steer left = positive
# angle left = negative
# left of the road = positive

class CrisisDriver(Driver):

    def __init__(self, logdata=True):
        """ Handles difficult situations in which the car gets stuck
        """
        super(CrisisDriver, self).__init__(logdata=logdata)
        self.iter = 0
        self.driver = Driver(logdata=False)
        self.is_in_control = False
        self.approaches = [self.navigate_to_middle, self.navigate_to_middle, self.original_implementation]
        self.previous_angles = []
        self.previous_accels = []
        self.previous_gears  = []
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
        command.accelerator = 0
        self.appr_counter += 1
        if self.appr_counter > APPROACH_TIMEOUT:
            self.next_approach()
        try:
            command = self.approaches[self.approach](carstate, command)
            self.previous_accels.append(command.accelerator)
            self.previous_gears.append(command.gear)
        except Exception as e:
            err(self.iter, "ERROR:", str(e))

        return command
    
    def pass_control(self, carstate):
        """ Initializes a new crisis that has not been handled yet

        Args:
            carstate:   The original carstate
        
        """
        err(self.iter,"CRISIS: control received")
        self.is_in_control = True
        self.approach = 0
        self.appr_counter = 0
        # check if car in front
        # check if car behind
        # check track angle and side of the road
        # determine reverse or straight ahead
    
    def return_control(self):
        """ Passes control back to the main driver """
        err(self.iter,"CRISIS: control returned")
        self.is_in_control = False
        self.needs_help = False
    
    def next_approach(self):
        """ Adjusts state to next approach """
        self.approach += 1
        self.appr_counter = 0
        if self.approach >= len(self.approaches):
            self.approach -= 1
            self.return_control()
        else:
            err(self.iter,"CRISIS: next approach:",
                self.approaches[self.approach].__name__)
    
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
        if len(self.previous_angles) >= MAX_ANGLES:
            self.previous_angles.pop(0)
        self.previous_angles.append(carstate.angle)
        if len(self.previous_accels) >= MAX_ACCELS:
            self.previous_accels.pop(0)
        self.previous_gears.append(sign(carstate.gear))
        if len(self.previous_gears)  >= MAX_GEARS:
            self.previous_gears.pop(0)

        self.is_on_left_side  = sign(carstate.distance_from_center) == DFC_L
        self.is_on_right_side = not self.is_on_left_side
        self.faces_left   = sign(carstate.angle) == ANG_L
        self.faces_right  = not self.faces_left
        self.faces_front  = abs(carstate.angle) < 90
        self.faces_back   = not self.faces_front
        self.faces_middle = self.is_on_left_side == self.faces_right

        self.is_standing_still = abs(carstate.speed_x) < 0.1 
        self.has_car_in_front  = car_in_front(carstate.opponents)
        self.has_car_behind    = car_behind(carstate.opponents)
        self.is_blocked        = blocked(self.previous_accels,
                                         self.previous_gears,
                                         carstate.speed_x)
        self.is_off_road = max(carstate.distances_from_edge) == -1
        self.is_reversed = self.faces_back
        self.is_going_in_circles = going_in_circles(self.previous_angles)
        self.is_traveling_sideways = abs(carstate.speed_y) > 15
        self.has_problem = self.is_off_road or \
                           self.is_going_in_circles or \
                           self.is_blocked or \
                           self.is_reversed or \
                           self.is_traveling_sideways
        if self.has_problem:
            self.bad_counter += 1
            if self.bad_counter >= BAD_COUNTER_THRESHOLD:
                if self.is_off_road:
                    debug(self.iter, "        off road")
                if self.is_going_in_circles:
                    debug(self.iter, "        going in circles")
                if self.is_blocked:
                    debug(self.iter, "        blocked")
                if self.is_reversed:
                    debug(self.iter, "        reversed")
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
        command = self.driver.drive(carstate) # TODO is this legal?
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
        debug(self.iter,"CRISIS: navigate_to_middle")

        dfc = carstate.distance_from_center
        if self.is_off_road:
            dist_spd = 2 if abs(dfc) > 10 else 1
            perp_angle = 90 * sign(dfc)
            if self.faces_middle:
                debug(self.iter,"        off road and facing road")
                diff_with_perp_angle = perp_angle - carstate.angle
                if not abs(diff_with_perp_angle) < PRP_ANG_MARGIN:
                    command.steering = sign(diff_with_perp_angle) * STR_R
                command.gear = 1
                command.accelerator = OFF_ROAD_ACC * dist_spd
            else:
                debug(self.iter,"        off road and not facing road")
                diff_with_perp_angle = perp_angle + carstate.angle
                if not abs(diff_with_perp_angle) < PRP_ANG_MARGIN:
                    command.steering = sign(diff_with_perp_angle) * STR_R
                command.gear = -1
                command.accelerator = OFF_ROAD_REV_ACC * dist_spd
        else:
            if abs(carstate.angle) < CTR_ANG_MARGIN:
                self.approach_succesful()
            else:
                if self.faces_middle or abs(carstate.angle) < 50: # TODO globalize
                    debug(self.iter,"        on road facing middle")
                    command.steering = to_ang(carstate.angle)
                    command.gear = 1
                    command.accelerator = ACC
                elif abs(dfc) > 0.5:
                    debug(self.iter,"        on road not facing middle")
                    command.steering = away_from_ang(carstate.angle)
                    command.gear = -1
                    command.accelerator = REV_ACC
        
        if self.is_blocked:
            debug(self.iter, "ang=%.2f, spd=%.2f, dfc=%.2f"
                      %(carstate.angle, carstate.speed_x, carstate.distance_from_center))
            debug(self.iter, "ste=%.2f, acc=%.2f, gea=%.2f"
                      %(command.steering, command.accelerator, command.gear))
        
        # when spinning out of control, don't do anything
        if self.is_traveling_sideways or self.is_going_in_circles:
            command.acceleration = 0
        
        # braking when shifting gear so it doesn't continue its course
        if any(not g == command.gear for g in self.previous_gears[-3:]):
            command.brake = 1
            command.acceleration = 0
        return command

#
# Problem Detectors
#
def car_in_front(opp):
    """ Returns True if there's a car in front of ours """
    return any([o < 1 for o in opp[17:18]])        # TODO are these the 20deg slices in front?

def car_behind(opp):
    """ Returns True if there's a car behind ours """
    return any([0 < 0 for o in [opp[0], opp[35]]])  # TODO and these in the back?

def going_in_circles(angles):
    """ Returns True if the car is rapidly going round in circles """
    # TODO checks if there is a pattern in the angles that
    # constitutes rotation. Take care of skip from 180 to -180.
    return False

def blocked(accels, gears, speed):
    """ Returns True if the car is blocked by something """
    same_gear = all([g > 0 for g in gears]) or all([g < 0 for g in gears])
    been_accel = all([a > 0 for a in accels])
    return been_accel and same_gear and abs(speed) < 1
