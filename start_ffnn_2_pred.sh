#! /usr/bin/env python3

from pytocl.main import main
from ffnn_driver_2_pred import FFNN__2_Pred_Client

if __name__ == '__main__':
    main(FFNN__2_Pred_Client(15, "ffnn_driver_2_pred.data"))
