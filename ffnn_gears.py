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
from collections import defaultdict


gear_number = 9 # 0 = reverse, 1 = neutral, 2=1st gear, etc. to 7th gear

class Gear_switcher(nn.Module):

	def __init__(self, hidden_dimension):
		super(Gear_switcher, self).__init__()
		n_states = 1 + gear_number
		n_actions = gear_number
		self.layer_1 = nn.Linear(n_states, hidden_dimension)
		self.layer_2 = nn.Linear(hidden_dimension, n_actions)

	def forward(self, inputs):
		out = self.layer_1(inputs)
		out = nn.functional.relu(out)
		out = self.layer_2(out)
		return out

def gear_to_tensor(gear_value):
	gear_value += 1
	return torch.LongTensor([gear_value])

def to_tensor(carstate):
	#gear, rpm
	return torch.FloatTensor(driver_support.binerize_input(value=carstate.gear, mapping=get_gear_map()) + [carstate.rpm])

def prediction_to_action(prediction):
	# the index is the gear
	index = prediction.data.numpy().argmax()
	index -= 1
	return index

def get_gear_map():
	gear_to_index_map = dict()
	for x in range(-1, gear_number - 1):
		gear_to_index_map[str(x)] = x + 1
	return gear_to_index_map

def evaluate(model, data):
	"""Evaluate a model on a data set."""
	correct = 0.0

	for y_true, state in data:
		y_true = int(y_true[0])
		lookup_tensor = Variable(torch.FloatTensor(state))
		scores = model(lookup_tensor)
		action = prediction_to_action(scores)

		if action == y_true:
			correct += 1

	print("percent correct={}".format(correct/len(data)))

def split_data_set(data_set, eval_perc=0.2):
	total = len(data_set)
	split = int(total*eval_perc)
	train = data_set[:split]
	evaluate = data_set[split:]
	return train, evaluate

def create_model(out_file, training_folder, learning_rate, epochs, hidden_dimension):
	# Read in the data
	training = []
	for file_in in [join(training_folder, f) for f in listdir(training_folder) if isfile(join(training_folder, f))]:
		training += list(driver_support.read_lliaw_dataset_gear_gear_rpm_spe(file_in))


	model = Gear_switcher(hidden_dimension)
	training = driver_support.binerize_data_input(data=training, index=0, mapping=get_gear_map())
	training, evalu = split_data_set(training)
	print(model)
	evaluate(model, evalu)
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)

	loss = nn.CrossEntropyLoss()

	for ITER in range(epochs):

		train_loss = 0.0
		start = time.time()
		lowest_gear = 10
		highest_gear = 0
		last_state = None

		for y_true, state in training:
			if last_state == None:
				last_state = state
				continue

			correct_gear = int(y_true[0])

			optimizer.zero_grad()

			in_state = Variable(torch.FloatTensor(last_state))
			y_pred = model(in_state).view(1, gear_number)
			y_true = Variable(gear_to_tensor(correct_gear))

			#print(y_true, prediction_to_action(y_pred))

			output = loss(y_pred, y_true)
			train_loss += output.data[0]

			# backward pass
			output.backward()

			# update weights
			optimizer.step()
			last_state = state

		print("last prediction made:pred={}, actual={}".format(prediction_to_action(y_pred), y_true))
		print("iter %r: train loss/action=%.4f, time=%.2fs" %(ITER, train_loss/len(training), time.time()-start))
	evaluate(model, evalu)
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
