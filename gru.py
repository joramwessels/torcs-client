# file:     gru.py
# author:   Joram Wessels
# date:     23-11-2017
# source:   https://github.com/spro/practical-pytorch
# description:
#
#       Trains a GRU RNN on your training data and saves the weights.
#       Run by calling "python gru.py <train_data_file> <save_file>".
#

"""
TODO not tested
TODO index out of range in GRU.forward()
"""

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import sys
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

allow_cuda = False
use_cuda = torch.cuda.is_available() and allow_cuda
teacher_forcing_ratio = 0.5

class GRU(nn.Module):

    def __init__(self, i_dim, h_dim, o_dim, n_layers=1):
        """ The encoder turns a continuous input into a continuous output

        i_dim:      The input dimension
        h_dim:      The amount of hidden units
        o_dim:      The output dimension
        n_layers:   The amount of hidden layers (default=1)

        """
        super(GRU, self).__init__()
        self.n_layers = n_layers
        self.h_dim = h_dim
        self.input_layer = nn.Linear(i_dim, h_dim)
        self.hidden_layer = nn.GRU(i_dim, h_dim, n_layers, batch_first=True)
        self.output_layer = nn.Linear(h_dim, o_dim)

    def forward(self, input, hidden):
        #input = input.view(1, 1, -1)
        #output = self.input_layer(input)
        #for i in range(self.n_layers):
        #    output, hidden = self.hidden_layer(output, hidden)
        print(input.size())
        print(hidden.size())
        output, hidden = self.hidden_layer(input, hidden)
        output = self.output_layer(output)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(self.n_layers, 1, self.h_dim))
        if use_cuda:
            return result.cuda()
        else:
            return result

def train(input, target, encoder, optimizer, criterion):
    """ Trains all timesteps for a sequence of parameters

    input:              Sequence of input variables
    target:             Sequence of target variables
    encoder:            The GRU encoder model
    optimizer:          The encoder optimizer object
    criterion:          The loss function

    """
    hidden = encoder.initHidden()
    optimizer.zero_grad()
    loss = 0
    #for i in range(len(input)):
    #    output, hidden = encoder(input[i], hidden)
    #    loss += criterion(output, target)
    output, hidden = encoder(input, hidden)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    return loss.data[0] / target.size()[0]

def train_iters(data, encoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    """ Trains the model for 'n_iters' epochs and reports on the progress

    data:           The training sequence pairs as a list of Variables of
                        floats, e.g [Variable(FloatTensor(FloatTensor))]
    encoder:        The encoder model
    n_iters:        The amount of sequences to train
    print_every:    The interval for printing the current loss
    plot_every:     The interval for adding a datapoint to the plot
    learning_rate:  The learning rate of the model

    """
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = data[iter - 1]
        input_variable = training_pair[0]
        target_variable = training_pair[1]
 
        loss = train(input_variable, target_variable, encoder, encoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)
    return 

def train_gru_on(filename, target_indices, save_as, sep=',', skip_first_line=True):
    """ Trains a GRU RNN on a given data file

    filename:           The name of a CSV file containing the training data
    target_indices:     A list of integers indicating the columns with target variables
    save_as:            Filename of the saved model parameter file
    sep:                The separator used in the CSV file
    skip_first_line:    Boolean indicating labels on the first line of the CSV

    """
    layers = 1
    units = 10
    iters = 500

    print("Reading data file",filename,"...")
    training_pairs = readTrainingData(filename, target_indices, sep, skip_first_line)
    print("Creating GRU model with %i layers of %i hidden units" %(layers, units),"...")
    model = GRU(len(training_pairs[0][0][0]), units, len(target_indices), n_layers=layers)
    print("Training model on data...")
    train_iters(training_pairs, model, iters)
    print("Model trained for %i iterations" %iters)
    model.save_state_dict(save_as)
    print("Model saved as",save_as)

def readTrainingData(filename, target_indices, sep, skip_first_line):
    """ Reads the data files to a list of sequence pairs

    filename:           The name of a CSV file containing the training data
    target_indices:     A list of integers indicating the columns with target variables
    sep:                The separator used in the CSV file
    skip_first_line:    Boolean indicating labels on the first line of the CSV

    """
    inputs = []
    targets = []
    with open(filename) as file:
        if skip_first_line: next(file)
        for line in file:
            params = [float(s) for s in line.strip().split(sep)]
            if not len(params) == 25: continue
            inputs.append(params)
            targets.append([params[i] for i in target_indices])
    if use_cuda:
        inputs = Variable(torch.FloatTensor([inputs])).cuda()
        targets = Variable(torch.FloatTensor([targets])).cuda()
    else:
        inputs = Variable(torch.FloatTensor([inputs]))
        targets = Variable(torch.FloatTensor([targets]))
    return [(inputs, targets)]

if __name__ == "__main__":
    train_gru_on(sys.argv[1], [0,1,2], sys.argv[2], sep=',', skip_first_line=True)