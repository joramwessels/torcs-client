#! /usr/bin/env python3

from pytocl.main import main
from ffnn_driver_2_pred import FFNN_2_Driver

if __name__ == '__main__':
    main(FFNN_2_Driver(15, "ffnn_driver_2_pred.data"))
