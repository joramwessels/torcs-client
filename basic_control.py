# file:         basic_control.py
# author:       Haukur Huppelepup
# date:         
# dependencies: 
# description:
# usage:
#

from math import radians

import numpy as np
import torch

import ffnn_speed
import ffnn_steer

ANGLES = [90, 75, 60, 45, 30, 20, 15, 10, 5, 0, -5, -10, -15, -20, -30, -45, -60, -75, -90]

class BasicControl:

    def __init__(self, steering_values):
        """ Short description

        Complete multiline
        description of class

        Args:
            steering_values:    ???
        
        """
        self.brake_row = 0
        self.speed = ffnn_speed.Speed(10)
        self.steer = ffnn_steer.Steer(10)
        self.steer.load_state_dict(torch.load("./steer.data"))
        self.speed.load_state_dict(torch.load("./ffnn_speed.data"))
        self.steering_values = steering_values
        self.alphas = [radians(x) for x in ANGLES]

    def deal_with_opponents(self, steer_pred, pedal, speed_x,
                        distance_from_center, opponents_new, opponents_delta):
        """ Description

        Args:
            steer_pred:             ?
            pedal:                  ?
            speed_x:                ?
            distance_from_center:   ?
            opponents_new:          ?
            opponents_delta:        ?
        
        Returns:
            ???
        
        """
        # index 18 is in front
        # index 35 in behind us
        adjustment = 0.1
        # if there are cars infront-left -> move to right
        if opponents_new[17] < 10 or opponents_new[16] < 10 or opponents_new[15] < 10:
            print("ADJUSTING SO NOT TO HIT")
            steer_pred -= adjustment
        if opponents_new[19] < 10 or  opponents_new[20] < 10 or opponents_new[21] < 10:
            print("ADJUSTING SO NOT TO HIT")
            # move to left
            steer_pred += adjustment
        if opponents_new[18] < 50:
            # we are on left side -> move right
            if distance_from_center > 0:
                steer_pred -= adjustment
            # o.w. move left
            else:
                steer_pred += adjustment
        if speed_x > 100:
            # we are getting closer to the car in front (and we can't avoid it). We need to slow down a bit
            if (opponents_delta[18] < 0 and opponents_new[18] < 20) or (opponents_delta[17] < 0 and opponents_new[17] < 4) or (opponents_delta[19] < 0 and opponents_new[19] < 4):
                pedal -= 0.1

        return steer_pred, pedal

    def steer_decider(self, carstate):
        """ Description

        Args:
            carstate:           The full carstate
        
        Returns:
            Steering angle?

        """
        alpha_index = np.argmax(carstate.distances_from_edge)
        if is_straight_line(carstate=carstate, radians=self.alphas[alpha_index], factor=self.steering_values[4]):
            return carstate.angle * 0.5

        steering_function = lambda index, offset:\
            (self.alphas[index-offset] * carstate.distances_from_edge[index-offset] \
            + self.alphas[index+offset] * carstate.distances_from_edge[index+offset]) \
            / (carstate.distances_from_edge[index+offset] \
              + carstate.distances_from_edge[index-offset])

        steer = self.steering_values[0] * self.alphas[alpha_index]
        for x in range(1, 4):
            if alpha_index - x > -1 and alpha_index + x < len(self.steering_values):
                steer += self.steering_values[x]*steering_function(alpha_index, x)
        return steer

    def speed_decider(self, carstate, max_speed=120):
        """ Description

        Args:
            carstate:           The full carstate
            max_speed:          ???
        
        Returns:
            ???

        """
        # we predict speed and map that to pedal
        x_in = ffnn_speed.carstate_to_variable(carstate)
        target_speed = self.speed(x_in).data[0]
        # we limit the speed
        if target_speed >= max_speed:
            target_speed = max_speed
        pedal = 2/(1 + np.exp(carstate.speed_x - target_speed))-1
        return pedal

    def gear_decider(self, carstate):
        """ Description

        Args:
            carstate:           The full carstate
        
        Returns:
            The gear to shift to (int)

        """
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

    def disambiguate_pedal(self, pedal, accel_cap=0.5, break_cap=0.75, break_max_length=5):
        """ Description

        Args:
            ???
        
        Returns:
            The break and accelerator command values

        """
        if pedal >= 0.0:
            accelerator = pedal*accel_cap
            brake = 0
        else:
            # we need to make sure that we don't break hard enough and not too long
            self.brake_row += 1
            if self.brake_row <= break_max_length:
                brake = abs(pedal)*break_cap
            else:
                self.brake_row = 0
                brake = 0
            accelerator = 0
        return brake, accelerator

def is_straight_line(carstate, radians, factor):
    """ Decides whether ??? is a straight line

    Args:
        carstate:           The full carstate
        radians:            ???
        factor:             ???
    
    Returns:
        A boolean indicating whether ???

    """
    if abs(carstate.distance_from_center) < 0.75:
        if radians == 0:
            return True
        if carstate.distances_from_edge[9] > 190:
            return True
        if carstate.distances_from_edge[9] > factor * carstate.speed_x:
            return True
    return False