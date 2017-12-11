# file:         crisis_driver.py
# authors:      Haukur, Joram Wessels
# date:         07-12-2017
# dependencies: pytocl, basic_control, swarm, crisis_driver
# description
# usage
#

from pytocl.driver import Driver
from pytocl.car import State, Command

from sys import stderr
from math import radians
from operator import sub

from driver_utils import *
from basic_control import BasicControl
from swarm import FeromoneTrail
from crisis_driver import CrisisDriver

# - NOTE crash detection for swarm only checks for off road
# - NOTE collision detection for swarm might be too sensitive
# - TODO clear up swarm global parameters
# - TODO swarm debug output is piped to stderr

ENABLE_SWARM = True
ENABLE_CRISIS_DRIVER = True

# swarm metaparameters
swarm_pos_int  = 50
swarm_spd_int  = 20
swarm_spd0     = 0
swarm_spd_n    = 20
swarm_expl_int = 40

class Final_Driver(Driver):

    def __init__(self, steering_values, global_max_speed):
        """ Short description

        Multiline description on
        details and usage

        Args:
            steering_values:    ???
            global_max_speed:   ???

        """
        super(Final_Driver, self).__init__()
        self.iter = 0
        self.basic_control = BasicControl(steering_values)
        self.back_up_driver = CrisisDriver(logdata=False)
        self.bad_counter = 0
        self.lap_counter = 0
        self.last_opponents = [0 for x in range(36)]
        self.global_max_speed = global_max_speed
        self.max_speed = global_max_speed
        self.cummulative_time = 0
        if ENABLE_SWARM:
            self.swarm = FeromoneTrail(
                swarm_pos_int, swarm_spd_int,
                swarm_spd0,    swarm_spd_n,
                swarm_expl_int, self.global_max_speed)
            self.crashed_in_last_frame = False
            self.contact_in_last_frame = False
            self.previous_frame_position = 0

    def drive(self, carstate: State) -> Command:
        """ Description

        Args:
            carstate:   All parameters packed in a State object

        Returns:
            command:    The next move packed in a Command object

        """
        self.iter += 1
        self.back_up_driver.update_status(carstate)

        # trackers
        self.update_trackers(carstate)
        if PRINT_STATE and (self.iter % PRINT_CYCLE_INTERVAL) == 0:
            self.print_trackers(carstate, r=True)

        # crash and collision detection for swarm
        if ENABLE_SWARM:
            if self.back_up_driver.needs_help:
                self.crashed_in_last_frame = True
            for dist in carstate.opponents:
                if dist == 0:
                    self.contact_in_last_frame = True

        # crisis handling
        if ENABLE_CRISIS_DRIVER:
            if self.back_up_driver.is_in_control:
                return self.back_up_driver.drive(carstate)
            elif self.back_up_driver.needs_help:
                err(self.iter, "SWARM:  crashed")
                self.back_up_driver.pass_control(carstate)
                return self.back_up_driver.drive(carstate)

        # since the data and python's values differ we need to adjust them
        carstate.angle   = radians(carstate.angle)
        carstate.speed_x = carstate.speed_x*3.6
        command = self.make_next_command(carstate)

        return command

    def make_next_command(self, carstate):
        """ Description

        Args:
            carstate:   The full carstate object as passed to Driver()

        Returns:
            command:    The command object to pass back to the server

        """

        # checking in on the swarm
        position = carstate.distance_from_start
        position = int(position - (position % self.swarm.pos_int))
        new_frame = position > (self.previous_frame_position + self.swarm.pos_int)
        new_lap = self.previous_frame_position > (position + self.swarm.pos_int)
        if ENABLE_SWARM and (new_frame or new_lap):
            self.max_speed = self.swarm.check_in(
                                        position,
                                        carstate.speed_x,
                                        self.crashed_in_last_frame,
                                        self.contact_in_last_frame)
            self.crashed_in_last_frame = False
            self.contact_in_last_frame = False
            self.previous_frame_position = position
            err(self.iter, "SWARM:  pos=%i, max_speed=%i" %(position, self.max_speed))

        # basic predictions
        gear       = self.basic_control.gear_decider(carstate)
        steer_pred = self.basic_control.steer_decider(carstate)
        pedal      = self.basic_control.speed_decider(carstate, max_speed=self.max_speed)

        # making sure we don't drive at people
        opponents_deltas = list(map(sub, carstate.opponents, self.last_opponents))
        steer_pred, pedal = self.basic_control.deal_with_opponents(steer_pred,
                                                              pedal,
                                                              carstate.speed_x,
                                                              carstate.distance_from_center,
                                                              carstate.opponents,
                                                              opponents_deltas)

        # disambiguating pedal with smoothing
        brake, accel = self.basic_control.disambiguate_pedal(pedal, accel_cap=1.0)

        # debug output
        if PRINT_COMMAND and self.iter % PRINT_CYCLE_INTERVAL:
            print("Executing comand: gear=%.2f, acc=%.2f," %(gear, accel),
                  "break=%.2f, steering=%.2f" %(brake, steer_pred))

        # command construction
        command = Command()
        command.brake = brake
        command.accelerator = accel
        command.steering = steer_pred
        command.gear = gear

        if command.steering > 0.15:
            debug(self.iter, "BASIC: turning right")
        elif command.steering < -0.15:
            debug(self.iter, "BASIC: turning left")

        return command

    def update_trackers(self, carstate):
        """ Updates info about the race """
        self.iter += 1
        if abs(carstate.current_lap_time) < 0.020:
            self.lap_counter += 1
            self.cummulative_time += carstate.last_lap_time + self.cummulative_time

    def print_trackers(self, carstate, r=False):
        """ Prints info on the race """
        line_end = '\r' if r else '\n'
        print("Lap=%i CurLapTime=%.2f dist=%.2f time=%.2f"
               %(self.lap_counter,
                 carstate.current_lap_time,               
                 carstate.distance_raced,
                 self.cummulative_time + carstate.current_lap_time)
               , end=line_end)
