from pytocl.driver import Driver
from pytocl.car import State, Command

import mlp

class MyDriver(Driver):
    def __init__(self, model_file="mlp_100x100.pt"):
        mlp.use_cuda = False
        self.model = mlp.load_model(model_file)
        self.it = 0

    def drive(self, carstate):
        self.it += 1

        x = [carstate.angle, carstate.speed_x,
            carstate.speed_y, carstate.speed_z] + \
            list(carstate.distances_from_edge) + \
            [carstate.distance_from_center]
        pred_y = list(self.model.predict(x).data)[0]
        command = Command()
        command.accelerator = pred_y[0]
        command.brake       = pred_y[1]
        command.steering    = pred_y[2]
        gear_flt = pred_y[3] if self.it > 750 else self.it/250.0
        command.gear = min(5, max(1, int(gear_flt + 0.5)))

        print(self.it,"acc: %.2f, brk: %.2f, ste: %.2f, gea: %.2f"
                       %(command.accelerator, command.brake,
                         command.steering, gear_flt), end='\r')
        return command
