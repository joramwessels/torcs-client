#! /usr/bin/env python3

from pytocl.main import main
from combined_driver import Final_Driver

steering = [0.19, 1.58, 0.69, 0.60, 2.01]
max_speed = 126

if __name__ == '__main__':
  main(Final_Driver(steering, max_speed))
