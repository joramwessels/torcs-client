from pytocl.driver import Driver
from pytocl.car import State, Command
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import time
import sys
from os import listdir
from os.path import isfile, join
import driver_support

class FFNN_Driver(Driver):

	def __init__(self, hidden_dimension, in_file):
		super(FFNN_Driver, self).__init__()
		self.model = FFNN(hidden_dimension)
		self.model.load_state_dict(torch.load(in_file))
		self.last_command = Command()
		self.last_command.accelerator = 1
		self.last_command.brake = 0
		self.last_command.steering = 0
		self.last_command.gear = 1
		self.sensor_data = []

	def drive(self, carstate: State) -> Command:
		# translate carstate to tensor for NN
		x_in = Variable(carstate_to_tensor(carstate))
		# get speed/steering target
		acc_break_steer_prediction = self.model(x_in).data
		print(acc_break_steer_prediction)

		# based on target, implement speed/steering manually
		command = Command()
		command.accelerator = acc_break_steer_prediction[0]
		command.brake = acc_break_steer_prediction[1]
		command.steering = acc_break_steer_prediction[2]
		command.gear = acc_break_steer_prediction[3]
		self.last_command = command
		return command

def carstate_to_tensor(carstate: State) -> torch.FloatTensor:
	# angle curLapTime distFromStartLine distRaced fuel gear racepos
	# rpm speedx speedy speedz tracksensor1_19 distToMiddle
	return torch.FloatTensor([carstate.angle,
								carstate.current_lap_time,
								carstate.distance_from_start,
								carstate.distance_raced,
								carstate.fuel,
								carstate.gear,
								carstate.race_position,
								carstate.rpm,
								carstate.speed_x,
								carstate.speed_y,
								carstate.speed_z] +
								list(carstate.distances_from_edge) +
								[carstate.distance_from_center]
								)


class FFNN(nn.Module):

	def __init__(self, hidden_dimension):
		super(FFNN, self).__init__()
		n_states = 31 # see above
		n_actions = 4 # acc + break + steering + gear
		self.layer_1 = nn.Linear(n_states, hidden_dimension)
		self.non_lin = nn.Sigmoid()
		self.layer_2 = nn.Linear(hidden_dimension, n_actions)

	def forward(self, inputs):
		out = self.layer_1(inputs)
		out = self.non_lin(out)
		out = self.layer_2(out)
		return out

def create_model(out_file, training_folder, learning_rate, epochs, hidden_dimension):
	epochs = int(epochs)
	learning_rate = float(learning_rate)
	hidden_dimension = int(hidden_dimension)
	# Read in the data
	training = []
	for file_in in [join(training_folder, f) for f in listdir(training_folder) if isfile(join(training_folder, f))]:
		training += list(driver_support.read_lliaw_dataset_acc_br_stee_ge(file_in))

	model = FFNN(hidden_dimension)
	print(model)
	optimizer = optim.SGD(model.parameters(), lr=learning_rate)

	# I'm not sure about this loss...
	loss = nn.MSELoss()

	for ITER in range(epochs):

		train_loss = 0.0
		start = time.time()

		for y_true, state in training:
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

		print("last prediction made:", y_pred, y_true)
		print("iter %r: train loss/action=%.4f, time=%.2fs" %(ITER, train_loss/len(training), time.time()-start))
	torch.save(model.state_dict(), out_file)

def main():
    create_model(*sys.argv[1:])

if __name__ == "__main__":
    main()
