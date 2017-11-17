from pytocl.driver import Driver
from pytocl.car import State, Command
import typing as t
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import time
import os

# Hyper Parameters
hidden_layer_size = 10
num_epochs = 5
learning_rate = 0.01
path = os.path.abspath(os.path.dirname(__file__))
save_to = os.path.join(path, "simple_nn.data")

class FFNNDriver(Driver):

	def __init__(self):
		super(FFNNDriver, self).__init__()
		self.model = SimpleNN()
		self.model.load_state_dict(torch.load(save_to))
		self.last_command = Command()
		self.last_command.accelerator = 1
		self.last_command.brake = 0
		self.last_command.steering = 0
		self.last_command.gear = 1
		#self.last_command.focus = 45
		self.sensor_data = []

	def drive(self, carstate: State) -> Command:
		# translate carstate to tensor for NN
		x_in = Variable(carstate_to_tensor(carstate))
		# get speed/steering target
		speed_steering_prediction = self.model(x_in).data
		print(speed_steering_prediction)
		steering_target, speed_target = speed_steering_prediction[0], speed_steering_prediction[1]

		# based on target, implement speed/steering manually
		command = Command()
		command.accelerator = self.get_accel(speed_target, carstate.speed_x)
		command.brake = self.get_break(speed_target, carstate.speed_x)
		command.steering = self.get_steer(target_pos=steering_target,
											actual_pos=self.last_command.steering,
											angle=carstate.angle)

		# gather data from field of vision
		command.gear = self.get_gear(carstate.rpm, carstate.gear)
		#command.focus = -self.last_command.focus
		self.last_command = command
		return command

	def get_gear(self, current_rpm, current_gear):
		print(current_rpm)
		if current_rpm >= 8500:
			current_gear += 1
		if current_rpm <= 2000:
			current_gear -= 1
		if current_gear <= 0:
			current_gear = 1
		if current_gear >= 6:
			current_gear = 6
		return current_gear

	def get_steer(self, target_pos, actual_pos, angle, epsilon=0.1):
		error = target_pos - actual_pos
		angle = angle + error * epsilon
		steer = angle/180
		return steer

	def get_accel(self, target_speed, actual_speed):
		accel = 0
		if (target_speed - actual_speed) > 0:
			accel = (target_speed - actual_speed)/20
		if target_speed - actual_speed > 20:
			accel = 1
		return accel

	def get_break(self, target_speed, actual_speed):
		brake = 0
		if (target_speed - actual_speed) < 0:
			brake = -(target_speed - actual_speed)/20
		if target_speed - actual_speed < -20:
			brake = 1
		return brake

def carstate_to_tensor(carstate: State) -> torch.FloatTensor:
	return torch.FloatTensor([carstate.race_position, carstate.angle] + list(carstate.distances_from_edge))

def read_dataset(filename: str) -> t.Iterable[t.Tuple[t.List[float], t.List[float]]]:
	with open(filename, "r") as f:
		next(f) # as the file is a csv, we don't want the first line
		# acc, break, steer,
		# 1.0, 0.0, -1.6378514771737686E-5,
		# speed, pos, angle,
		# -0.0379823,-5.61714E-5,4.30409E-4,
		# distances... len(distances) = 19
		# 5.00028, 5.0778, 5.32202, 5.77526, 6.52976, 7.78305, 10.008, 14.6372, 28.8659, 200.0,
		# 28.7221, 14.6009, 9.99199, 7.7742, 6.52431, 5.77175, 5.31976, 5.07646, 4.99972
		for line in f:
			# we predict based on steer + speed
			# based on pos + angle + distances
			yield ([float(x) for x in line.strip().split(",")[2:4]], [float(x) for x in line.strip().split(",")[4:]])


class SimpleNN(nn.Module):

	def __init__(self):
		super(SimpleNN, self).__init__()
		n_states = 21 # track_pos + angle + distances
		n_actions = 2 # speed + steering
		self.layer_1 = nn.Linear(n_states, hidden_layer_size)
		self.relu = nn.ReLU()
		self.layer_2 = nn.Linear(hidden_layer_size, n_actions)

	def forward(self, inputs):
		out = self.layer_1(inputs)
		out = self.relu(out)
		out = self.layer_2(out)
		return out

def create_model():
	# Read in the data
	save_to = os.path.join(path, "simple_nn.data")
	aalborg = list(read_dataset(os.path.join(path, "train_data/aalborg.csv")))
	alpine_1 = list(read_dataset(os.path.join(path, "train_data/alpine-1.csv")))
	speedway = list(read_dataset(os.path.join(path, "train_data/f-speedway.csv")))
	train = aalborg + alpine_1 + speedway

	model = SimpleNN()
	print(model)
	optimizer = optim.SGD(model.parameters(), lr=learning_rate)

	# I'm not sure about this loss...
	loss = nn.MSELoss()

	for ITER in range(num_epochs):

		train_loss = 0.0
		start = time.time()

		# there is one state which is "broken" as it only contains 18 values. Therefore, use try-except
		try:

			for y_true, state in train:
				# forward pass
				in_state = Variable(torch.FloatTensor(state))
				y_pred = model(in_state)
				y_true = Variable(torch.FloatTensor(y_true))
				output = loss(y_pred, y_true)
				train_loss += output.data[0]

				# backward pass
				optimizer.zero_grad()
				output.backward()

				# update weights
				optimizer.step()
		except:
			print("error:", in_state)

		print("iter %r: train loss/sent=%.4f, time=%.2fs" %(ITER, train_loss/len(train), time.time()-start))
	torch.save(model.state_dict(), save_to)

def main():
    create_model()

if __name__ == "__main__":
    main()
