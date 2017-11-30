import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.optim as optim
import argparse
import driver_support
import time
import sys
from os import listdir
from os.path import isfile, join

class RNNMove(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, input_dimension, hidden_dimension, output_dimension, nlayers, dropout=0.5, tie_weights=False):
        super(RNNMove, self).__init__()
        self.drop = nn.Dropout(dropout)
        nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
        self.rnn = nn.RNN(input_dimension, hidden_dimension, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(hidden_dimension, output_dimension)

        self.init_weights()

        self.rnn_type = rnn_type
        self.hidden_dimension = hidden_dimension
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        return nn.Parameter(torch.zeros(self.nlayers, bsz, self.hidden_dimension), requires_grad=True)

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
    	training += list(driver_support.read_lliaw_dataset_acc_bre_steer_bunch(file_in))

    n_states = 22
    n_actions = 3
    n_layers = 1

    model = RNNMove("RNN_RELU", n_states, hidden_dimension, n_actions, n_layers, dropout=0.5, tie_weights=False)
    training, evalu = split_data_set(training)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss = nn.MSELoss()

    for ITER in range(epochs):

        train_loss = 0.0
        start = time.time()
        hidden = model.init_hidden(1)

        for y_true, state in training:
            optimizer.zero_grad()

            in_state = Variable(torch.FloatTensor(state))
            y_pred, hidden = model(in_state, hidden)
            y_true = Variable(torch.FloatTensor(y_true))

            #print(y_true, prediction_to_action(y_pred))

            output = loss(y_pred, y_true)
            train_loss += output.data[0]

            # backward pass
            output.backward()

            # update weights
            optimizer.step()

        print("last prediction made:pred={}, actual={}".format(prediction_to_action(y_pred), y_true))
        print("iter %r: train loss/action=%.4f, time=%.2fs" %(ITER, train_loss/len(training), time.time()-start))
    #evaluate(model, evalu)
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
