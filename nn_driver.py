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
hidden_size = 10
num_epochs = 5
learning_rate = 0.01
n_actions = 3
n_states = 3
path = os.path.abspath(os.path.dirname(__file__))
save_to = os.path.join(path, "simple_nn.data")

class FFNNDriver(Driver):

	def __init__(self):
		super(FFNNDriver, self).__init__()
		self.model = SimpleNN(n_states, hidden_size, n_actions)
		self.model.load_state_dict(torch.load(save_to))
		self.last_command = None
		self.sensor_data = []

	def drive(self, carstate: State) -> Command:
		x_in = Variable(carstate_to_tensor(carstate))
		out_command = self.model(x_in).data
		command = Command()
		command.accelerator = out_command[0]
		command.brake = out_command[1]
		command.steering = out_command[2]
		command.gear = 1
		if not self.last_command or self.last_command.focus == -45:
			command.focus = 45
		else:
			command.focus = -45
		self.last_command = command
		return command

	def get_steer(self, target_pos, actual_pos, epsilon=0.01):
		angle = carstate.angle
		error = target_pos - actual_pos
		angle = angle + error * epsilon
		steer = angle/SteerLock
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
	if carstate.focused_distances_from_egde_valid:
		print(carstate.focused_distances_from_edge)
	return torch.FloatTensor([carstate.speed_x, carstate.race_position, carstate.angle])

def read_dataset(filename: str) -> t.Iterable[t.Tuple[t.List[float], t.List[float]]]:
	with open(filename, "r") as f:
		next(f) # as the file is a csv, we don't want the first line
		for line in f:
			yield ([float(x) for x in line.strip().split(",")[0:3]], [float(x) for x in line.strip().split(",")[3:6]])


class SimpleNN(nn.Module):

	def __init__(self, input_d, hidden_d, out_d):
		super(SimpleNN, self).__init__()
		self.layer_1 = nn.Linear(input_d, hidden_d)
		self.relu = nn.ReLU()
		self.layer_2 = nn.Linear(hidden_d, out_d)

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

	model = SimpleNN(n_states, hidden_size, n_actions)
	print(model)
	optimizer = optim.SGD(model.parameters(), lr=learning_rate)

	# I'm not sure about this loss...
	loss = nn.MSELoss()

	for ITER in range(num_epochs):

		train_loss = 0.0
		start = time.time()

		# there is one state which is "broken" as it only contains 18 values. Therefore, use try-except
		try:

			for command, state in train:
				# forward pass
				in_state = Variable(torch.FloatTensor(state))
				y_pred = model(in_state)
				y = Variable(torch.FloatTensor(command))
				output = loss(y_pred, y)
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
