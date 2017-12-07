from pytocl.driver import Driver
from pytocl.car import State, Command
import time
from math import radians
import driver_support
from operator import sub

from basic_control import BasicControl
from swarm import FeromoneTrail

# - NOTE crash detection for swarm only checks for off road
# - NOTE collision detection for swarm might be too sensitive
# - TODO clear up swarm global parameters

# swarm metaparameters
swarm_pos_int  = 20
swarm_spd_int  = 10
swarm_spd0     = 100
swarm_spd_n    = 30
swarm_expl_int = 50

class Final_Driver(Driver):

    def __init__(self, steering_values, global_max_speed):
        super(Final_Driver, self).__init__()
        self.basic_control = BasicControl(steering_values)
        self.back_up_driver = Driver(logdata=False)
        self.bad_counter = 0
        self.lap_counter = 0
        self.last_opponents = [0 for x in range(36)]
        self.global_max_speed = global_max_speed
        self.max_speed = global_max_speed
        self.cummulative_time = 0
        self.swarm = FeromoneTrail(
            swarm_pos_int, swarm_spd_int,
            swarm_spd0,    swarm_spd_n,
            swarm_expl_int, global_max_speed)
        self.crashed_in_last_frame = False
        self.contact_in_last_frame = False
        self.previous_frame_position = 0

    def update_trackers(self, carstate):
        print(carstate.current_lap_time)
        if abs(carstate.current_lap_time) < 0.020:
            self.lap_counter += 1
            print("Lap={}".format(self.lap_counter))
            self.cummulative_time += carstate.last_lap_time + self.cummulative_time

        print(carstate.angle)
        print(carstate.distance_from_center)
        print("distance={}".format(carstate.distance_raced))
        print("time={}".format(self.cummulative_time + carstate.current_lap_time))

    def drive(self, carstate: State) -> Command:
        self.update_trackers(carstate)

        # crash and collision detection for swarm
        if is_offroad(carstate):
            self.crashed_in_last_frame = True
        for dist in carstate.opponents:
            if dist == 0:
                self.contact_in_last_frame = True

        # bad state detection
        if in_a_bad_place(carstate, self.bad_counter):
            command = self.back_up_driver.drive(carstate)
            if self.bad_counter >= 600 and is_stuck(carstate):
                # we try reversing
                command.gear = -command.gear
                if command.gear < 0:
                    command.steering = -command.steering
                    command.gear = -1
                self.bad_counter = 200
        else:
            # since the data and python's values differ we need to adjust them
            carstate.angle = radians(carstate.angle)
            carstate.speed_x = carstate.speed_x*3.6
            command = self.make_next_command(carstate)
            # based on target, implement speed/steering manually
        print("Executing command: gear={}, acc={}, break={}, steering={}".format(command.gear,
                                                                                 command.accelerator,
                                                                                 command.brake,
                                                                                 command.steering))

        return command

    def make_next_command(self, carstate):
        """ Description

        Args:
            carstate:   The full carstate object as passed to Driver()

        Returns:
            command:    The command object to pass back to the server

        """
        command = Command()

        # we switch gears manually
        gear = self.basic_control.gear_decider(carstate)
        # we get the steering prediction
        steer_pred = self.basic_control.steer_decider(carstate)
        # checking in on the swarm
        position = carstate.distance_from_start
        if position > self.previous_frame_position + self.swarm.pos_int:
            print(position, self.previous_frame_position)
            position = int(position - (position % self.swarm.pos_int))
            self.max_speed = self.swarm.check_in(
                                        position,
                                        carstate.speed_x,
                                        self.crashed_in_last_frame,
                                        self.contact_in_last_frame)
            self.crashed_in_last_frame = False
            self.contact_in_last_frame = False
            self.previous_frame_position = position

        pedal = self.basic_control.speed_decider(carstate, max_speed=self.max_speed)

        # make sure we don't drive at people
        opponents_deltas = list(map(sub, carstate.opponents, self.last_opponents))
        steer_pred, pedal = self.basic_control.deal_with_opponents(steer_pred,
                                                              pedal,
                                                              carstate.speed_x,
                                                              carstate.distance_from_center,
                                                              carstate.opponents,
                                                              opponents_deltas)
        # disambiguate pedal with smoothing
        brake, accel = self.basic_control.disambiguate_pedal(pedal, accel_cap=1.0)

        command.brake = brake
        command.accelerator = accel
        command.steering = steer_pred
        command.gear = gear
        return command

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
        print("I'm offroad!")
        something_wrong = True
    if is_reversed(carstate):
        print("I'm reversed!")
        something_wrong = True
    if is_stuck(carstate):
        print("I'm stuck!")
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

def is_stuck(carstate):
    """ Returns True if car is stuck """
    return abs(carstate.speed_x) <= 5 and carstate.current_lap_time >= 10

def is_reversed(carstate):
    """ Returns True if the car is facing the wrong side """
    return abs(carstate.angle) >= 90
