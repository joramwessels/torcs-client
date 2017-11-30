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
        self.gears = ffnn_gears.Gear_switcher(15)
        self.gears.load_state_dict(torch.load("./gear.data"))
        self.move = ffnn_move.Steer_Acc_Break(25)
        self.move.load_state_dict(torch.load("./move.data"))


    def drive(self, carstate: State) -> Command:
        # translate carstate to tensor for NN
        x_in_move = Variable(ffnn_move.carstate_to_tensor(carstate))
        # get speed/steering target
        accel_pred, break_pred, steer_pred = self.move(x_in_move).data
        print(accel_pred, break_pred, steer_pred)
        x_in_gear = Variable(ffnn_gears.to_tensor(accel_pred, break_pred, carstate))
        gear = ffnn_gears.prediction_to_action(self.gears(x_in_gear))
        print(gear)

        if abs(carstate.speed_x) <= 10:
            gear = 1

        # based on target, implement speed/steering manually
        print("Executing: gear={}, acc={}, break={}, steering={}".format(gear, accel_pred, break_pred, steer_pred))
        command = Command()
        command.accelerator = accel_pred
        command.brake = break_pred
        command.steering = steer_pred
        command.gear = gear
        return command
