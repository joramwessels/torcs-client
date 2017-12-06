#! /usr/bin/env python3

from pytocl.main import main
from combined_driver import Final_Driver

steering = [REPLACE_STEERING]
max_speed = REPLACE_MAX_SPEED

if __name__ == '__main__':
  main(Final_Driver(steering, max_speed))
