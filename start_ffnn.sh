#! /usr/bin/env python3

from pytocl.main import main
from combined_driver import Final_Driver

steering = [0.21, 1.56, 0.68, 0.53, 1.25]
max_speed = 160

if __name__ == '__main__':
  main(Final_Driver(steering, max_speed))
