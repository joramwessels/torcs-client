from pytocl.driver import Driver
from pytocl.car import State, Command


class MyDriver(Driver):
	def drive(self, carstate):
		command = Command()
		command.steering = carstate.angle / 180
		if carstate.angle > 30 or carstate.angle < -30:
			command.brake = 0.5
			command.accelerator = 0
		else:
			command.brake = 0
			command.accelerator = 1
		command.gear = 1
		return command
