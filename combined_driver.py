from pytocl.driver import Driver
from pytocl.car import State, Command
import torch
import time
import sys
import math
import numpy as np
from os import listdir
from os.path import isfile, join
import driver_support
from torch.autograd import Variable

import ffnn_steer
import ffnn_speed

class Final_Driver(Driver):

    def __init__(self):
        super(Final_Driver, self).__init__()
        self.steer = ffnn_steer.Steer(10)
        self.steer.load_state_dict(torch.load("./steer.data"))
        self.speed = ffnn_speed.Speed(10)
        self.speed.load_state_dict(torch.load("./ffnn_speed.data"))
        self.back_up_driver = Driver(logdata=False)
        self.bad_counter = 0
        self.lap_counter = 0
        self.brake_row = 0
        self.angles = [90, 75, 60, 45, 30, 20, 15, 10, 5, 0, -5, -10, -15, -20, -30, -45, -60, -75, -90]
        self.alphas = [math.radians(x) for x in self.angles]

    def update_trackers(self, carstate):
        if carstate.laptime == 0:
            self.lap_counter += 1
            print("Lap={}".format(self.lap_counter))

    def drive(self, carstate: State) -> Command:

        if self.in_a_bad_place(carstate):
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
            carstate.angle = math.radians(carstate.angle)
            carstate.speed_x = carstate.speed_x*3.6
            command = self.make_next_command(carstate)
            # based on target, implement speed/steering manually
        print("Executing command: gear={}, acc={}, break={}, steering={}".format(command.gear,
                                                                                 command.accelerator,
                                                                                 command.brake,
                                                                                 command.steering))

        return command

    def make_next_command(self, carstate):
        command = Command()
        # we switch gears manually
        gear = self.gear_decider(carstate)
        # we get the steering prediction
        steer_pred = self.steer_decider(carstate, [0.21, 1.56, 0.68, 0.53, 1.25])
        # steer_pred = self.steer_decider_nn(carstate)
        # pedal =[-1;1], combining breaking and accelerating to one variable
        pedal = self.speed_decider(carstate)
        if pedal >= 0.0:
            command.accelerator = pedal*0.50
            command.brake = 0
        else:
            # we need to make sure that we don't break hard enough and not too long
            self.brake_row += 1
            if self.brake_row <= 5:
                command.brake = abs(pedal)*0.75
            else:
                self.brake_row = 0
                command.brake = 0
            command.accelerator = 0
        command.steering = steer_pred
        command.gear = gear
        return command

    def steer_decider_nn(self, carstate):
        x_in = ffnn_steer.carstate_to_variable(carstate)
        steer_pred = self.steer(x_in).data[0]
        return steer_pred

    def steer_decider(self, carstate, steering_values):
        alpha_index = np.argmax(carstate.distances_from_edge)
        if is_straight_line(carstate=carstate, radians=self.alphas[alpha_index], factor=steering_values[4]):
            return math.radians(carstate.angle)*0.5

        steering_function = lambda index, offset: (self.alphas[index-offset]*carstate.distances_from_edge[index-offset] + self.alphas[index+offset]*carstate.distances_from_edge[index+offset])/(carstate.distances_from_edge[index+offset]+carstate.distances_from_edge[index-offset])

        steer = steering_values[0]*self.alphas[alpha_index]
        steer += steering_values[1]*steering_function(alpha_index, 1)
        steer += steering_values[2]*steering_function(alpha_index, 2)
        steer += steering_values[3]*steering_function(alpha_index, 3)
        return steer


    def speed_decider(self, carstate):
        # we predict speed and map that to pedal
        x_in = ffnn_speed.carstate_to_variable(carstate)
        target_speed = self.speed(x_in).data[0]
        # we limit the speed
        if target_speed >= 120:
            print(target_speed)
            print(carstate.speed_x)
            target_speed = 120
        pedal = 2/(1 + np.exp(carstate.speed_x - target_speed))-1
        print(pedal)
        return pedal

    def gear_decider(self, carstate):
        gear = carstate.gear
        rpm = carstate.rpm
        # we do gears by hand
        # up if {9500 9500 9500 9500 9000}
        # down if {4000 6300 7000 7300 7300}
        if gear == -1:
            return 1
        elif gear == 0:
            if rpm >= 5000:
                gear = 1
        elif gear == 1:
            if rpm >= 9500:
                gear = 2
        elif gear == 2:
            if rpm >= 9500:
                gear = 3
            elif rpm <= 4000:
                gear = 2
        elif gear == 3:
            if rpm >= 9500:
                gear = 4
            elif rpm <= 6300:
                gear = 3
        elif gear == 4:
            if rpm >= 9500:
                gear = 5
            elif rpm <= 7000:
                gear = 3
        elif gear == 5:
            if rpm >= 9000:
                gear = 6
            elif rpm <= 7300:
                gear = 4
        elif gear == 6:
            if rpm <= 7300:
                gear = 5

        return gear


    def in_a_bad_place(self, carstate):
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
            self.bad_counter += 1
        else:
            self.bad_counter = 0
        # if we have been in a bad place for 2 seconds
        if self.bad_counter >= 100:
            return True
        return False

def is_straight_line(carstate, radians, factor):
    if abs(carstate.distance_from_center) < 0.75:
        if radians == 0:
            return True
        if carstate.distances_from_edge[9] > 190:
            return True
        if carstate.distances_from_edge[9] > factor * carstate.speed_x:
            return True
    return False


def is_offroad(carstate):
    return max(carstate.distances_from_edge) == -1

def is_stuck(carstate):
    return abs(carstate.speed_x) <= 5 and carstate.current_lap_time >= 10

def is_reversed(carstate):
    return abs(carstate.angle) >= 90
