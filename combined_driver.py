from pytocl.driver import Driver
from pytocl.car import State, Command
import torch
import time
import sys
from os import listdir
from os.path import isfile, join
import driver_support
from torch.autograd import Variable

import ffnn_gears
import ffnn_move

class Final_Driver(Driver):

    def __init__(self):
        super(Final_Driver, self).__init__()
        self.gears = ffnn_gears.Gear_switcher(20)
        self.gears.load_state_dict(torch.load("./gear.data"))
        self.move = ffnn_move.Steer_Acc_Break(25)
        self.move.load_state_dict(torch.load("./move.data"))
        self.back_up_driver = Driver(logdata=False)
        self.bad_counter = 0


    def drive(self, carstate: State) -> Command:
        if self.in_a_bad_place(carstate):
            command = self.back_up_driver.drive(carstate)
        else:
            command = self.make_next_command(carstate)
            # based on target, implement speed/steering manually
        print("Executing command: gear={}, acc={}, break={}, steering={}".format(command.gear,
                                                                                 command.accelerator,
                                                                                 command.brake,
                                                                                 command.steering))

        return command

    def make_next_command(self, carstate):
        # translate carstate to tensor for NN
        x_in_move = Variable(ffnn_move.carstate_to_tensor(carstate))
        # get speed/steering target
        accel_pred, break_pred, steer_pred = self.move(x_in_move).data
        x_in_gear = Variable(ffnn_gears.to_tensor(carstate))
        gear = ffnn_gears.prediction_to_action(self.gears(x_in_gear))
        command = Command()
        command.accelerator = accel_pred
        command.brake = break_pred
        command.steering = steer_pred
        command.gear = gear
        return command

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

def is_offroad(carstate):
    return max(carstate.distances_from_edge) == -1

def is_stuck(carstate):
    return abs(carstate.speed_x) <= 5 and carstate.current_lap_time >= 10

def is_reversed(carstate):
    return abs(carstate.angle) >= 90
