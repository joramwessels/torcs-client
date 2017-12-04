from pytocl.driver import Driver
from pytocl.car import State, Command

import mlp

class MyDriver(Driver):
    def __init__(self, model_file="mlp_model_1.pt"):
        mlp.use_cuda = False
        self.model = mlp.load_model(model_file)

    def drive(self, carstate):
        x = [carstate.angle, carstate.speed_x,
            carstate.speed_y, carstate.speed_z] + \
            list(carstate.distances_from_edge) + \
            [carstate.distance_from_center]
        pred_y = self.model.predict(x)
        command = Command()
        command.accelerator = pred_y[0]
        command.brake       = pred_y[1]
        command.steering    = pred_y[2]
        command.gear        = int(pred_y[3] + 0.5)
        return command