#! /usr/bin/env python3

from pytocl.main import main
from my_driver import MyDriver
from nn_driver import FFNNDriver

if __name__ == '__main__':
    main(FFNNDriver())
