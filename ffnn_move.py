import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import sys
import numpy as np
import argparse
import driver_support
from os import listdir
from os.path import isfile, join

class Steer_Acc_Break(nn.Module):

	def __init__(self, hidden_dimension):
		super(Steer_Acc_Break, self).__init__()
		n_states = 22
		n_actions = 3
		self.layer_1 = nn.Linear(n_states, hidden_dimension)
		self.non_lin = nn.Sigmoid()
		self.layer_2 = nn.Linear(hidden_dimension, n_actions)

	def forward(self, inputs):
		out = self.layer_1(inputs)
		out = self.non_lin(out)
		out = self.layer_2(out)
		return out

def carstate_to_tensor(carstate):
	# y=accel, break, stear, x=angle, speed, distance*19, distToMiddle
	return torch.FloatTensor([carstate.angle, carstate.speed_x] + list(carstate.distances_from_edge) + [carstate.distance_from_center])

def create_model(out_file, training_folder, learning_rate, epochs, hidden_dimension):
	# Read in the data
	training = []
	for file_in in [join(training_folder, f) for f in listdir(training_folder) if isfile(join(training_folder, f))]:
		training += list(driver_support.read_lliaw_dataset_acc_bre_steer_bunch(file_in))

	model = Steer_Acc_Break(hidden_dimension)
	print(model)
	optimizer = optim.SGD(model.parameters(), lr=learning_rate)

	loss = nn.MSELoss()

	for ITER in range(epochs):

		train_loss = 0.0
		start = time.time()

		for y_true, state in training:
			# forward pass
			optimizer.zero_grad()

			in_state = Variable(torch.FloatTensor(state))
			y_pred = model(in_state)
			y_true = Variable(torch.FloatTensor(y_true))

			#print(y_true, y_pred)

			output = loss(y_pred.view(1, 3), y_true)
			train_loss += output.data[0]

			# backward pass
			output.backward()

			# update weights
			optimizer.step()

		print("last prediction made:", y_pred, y_true)
		print("iter %r: train loss/action=%.4f, time=%.2fs" %(ITER, train_loss/len(training), time.time()-start))
	torch.save(model.state_dict(), out_file)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--epochs', type=int)
	parser.add_argument('--hidden', type=int)
	parser.add_argument('--learn', type=float)
	parser.add_argument('--in_file', type=str)
	parser.add_argument('--out_file', type=str)
	args = parser.parse_args()
	create_model(args.out_file, args.in_file, args.learn, args.epochs, args.hidden)

if __name__ == "__main__":
    main()
