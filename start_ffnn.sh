#! /usr/bin/env python3

from pytocl.main import main
from ffnn_driver import FFNN_Driver

if __name__ == '__main__':
    main(FFNN_Driver(15, "ffnn.data"))
