# file:     mlp.py
# author:   Joram Wessels
# source:   http://pytorch.org/tutorials/beginner/
# date:     15-11-2017
# description:
#       Trains a fully connected feed forward neural network with
#       a variable amount of units and layers. The model can be
#       trained using CUDA, saved, loaded, and used without CUDA
#       by calling model.predict([v1, v2, ..., vn])

# TODO Every training batch shows the same sequence of losses
#      There's no improvement among batches

import sys, os
import torch
from torch.autograd import Variable

learning_rate = 5e-7
epochs = 10
layers = 100
units = 100
allow_cuda = True
use_cuda = torch.cuda.is_available() and allow_cuda

class MLP(torch.nn.Module):

    def __init__(self, D_inp, D_hid, D_out, layers, x_scale, y_scale):
        """ Multilayer Perceptron with a variable amount of layers

        Args:
            D_inp:      Dimension of the input layer
            D_hid:      Dimension of all hidden layers
            D_out:      Dimension of the output layer
            layers:     The total amount of layers (2 means 1 hidden layer)
            x_scale:    The maximum values for all input variables
            y_scale:    The maximum values for all target variables

        """
        super(MLP, self).__init__()
        self.input_layer  = torch.nn.Linear(D_inp, D_hid)
        self.hidden_layer = torch.nn.Linear(D_hid, D_hid)
        self.output_layer = torch.nn.Linear(D_hid, D_out)
        self.layers = layers
        self.sigmoid = torch.nn.Sigmoid()
        self.x_scale = x_scale.cuda() if use_cuda else x_scale
        self.y_scale = y_scale.cuda() if use_cuda else y_scale
    
    def forward(self, x):
        """ Forward propagation: call to predict using current weights

        Args:
            x:  Input tensor as a 'D_inp'-dimensional torch.autograd.Variable

        Returns:
            The predicted target value given this input
        
        """
        h = self.input_layer(x)
        h = self.sigmoid(h)
        for _ in range(self.layers-1):
            h = self.hidden_layer(h)
            h = self.sigmoid(h)
        y_pred = self.output_layer(h)
        return y_pred.cuda() if use_cuda else y_pred
    
    def predict(self, x):
        """ Predicts a single variable, given as a list

        Args:
            x:  The input variables, given as a normal list
        
        Returns:
            The prediction as a list
        
        """
        x_var = Variable(torch.FloatTensor([x]), requires_grad=False)
        if use_cuda: x_norm = x_var.cuda() / self.x_scale
        else: x_norm = x_var / self.x_scale
        pred_var = self.forward(x_norm)[0]
        pred_var = pred_var * self.y_scale
        return pred_var

def train_model(x, y, metaparams):
    """ Trains a fully connected feed forward network

    Args:
        x:     An autograd Variable with inputs (batch, n_samples, n_var)
        y:     An autograd Variable with targets (batch, n_samples, n_var)
        metaparams:     A dictionary including the fields
            d_inp:      Dimension of the input layer
            d_hid:      Dimension of all hidden layers
            d_out:      Dimension of the output layer
            layers:     The total amount of layers (2 means 1 hidden layer)
            x_max:    The maximum values for all input variables
            y_max:    The maximum values for all target variables
    
    Returns:
        A torch model object
    
    """
    model = MLP(metaparams['d_inp'], metaparams['d_hid'],
                metaparams['d_out'], metaparams['layers'],
                metaparams['x_max'], metaparams['y_max'])
    if use_cuda: model = model.cuda()
    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print("Batch",t)
        for b in range(len(x)):

            # Forward
            y_pred = model(x[b])
            loss = criterion(y_pred, y[b])
            #print(t, loss.data[0])
            print("Training loss:",loss.data[0])

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return model

def read_file(filename, tgt_ind, inp_ind, sep=',', skip_first_line=False):
    """ Reads the CSV training data

    Args:
        filename:           The name of the CSV data file
        tgt_ind:            A list of indices indicating the targets
        inp_ind:            A list of indices indicating the inputs
        sep:                The separator used in the CSV
        skip_first_line:    Boolean indicating labels on the first line
    
    Returns:
        (x, y) tuple with 2D torch autograd Variables
    
    """
    x = []
    y = []
    with open(filename) as file:
        if skip_first_line: next(file)
        for line in file:
            clean_line = line.strip().split(sep)
            if len(clean_line) > 1:
                params = [float(s) for s in clean_line if not s == '']
                x.append([params[i] for i in inp_ind])
                y.append([params[i] for i in tgt_ind])
    x = Variable(torch.FloatTensor(x))
    y = Variable(torch.FloatTensor(y), requires_grad=False)
    return (x.cuda(), y.cuda()) if use_cuda else (x, y)

def read_all_files(folder, tgt_ind, inp_ind):
    """ Reads every file into a batch of the data

    Args:
        folder:     The path to the folder with the data files
        tgt_ind:    A list of indices indicating the targets
        inp_ind:    A list of indices indicating the inputs
    
    Returns:
        Two lists of autograd Variables (batch, n_samples, n_var)
    
    """
    xs, ys = [], []
    for f in os.listdir(folder):
        file = os.path.join(folder, f)
        if os.path.isfile(file):
            x, y = read_file(file, tgt_ind, inp_ind)
            xs.append(x)
            ys.append(y)
    return xs, ys

def load_model(filename, cuda=False):
    """ Loads a model from just the filename

    Args:
        filename:   The name of the save file
        cuda:       Set to True to make predictions use CUDA

    Returns:
        model

    """
    global use_cuda
    use_cuda = cuda
    metaparams = torch.load(filename + ".meta")
    model = MLP(metaparams['d_inp'], metaparams['d_hid'],
                metaparams['d_out'], metaparams['layers'],
                metaparams['x_max'], metaparams['y_max'])
    model.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
    return model.cuda() if use_cuda else model

def normalize(x, y, metaparams):
    """ Normalizes the data by dividing all variables by their max

    Args:
        x:      The x data (batch, n_samples, n_vars)
        y:      The y data (batch, n_samples, n_vars)
        metaparams:     The dictionary of metaparameters containing the max
    
    Returns:
        A tuple with the normalized x, y Variables
    
    """
    x_max, y_max = metaparams['x_max'], metaparams['y_max']
    if use_cuda: x_max, y_max = x_max.cuda(), y_max.cuda()
    y_max[y_max==0] = 1     # to prevent
    x_max[x_max==0] = 1     # division by zero
    for i in range(len(x)):
        for j in range(len(x[i])):
            x[i][j] = torch.div(x[i][j], x_max)
            y[i][j] = torch.div(y[i][j], y_max)
    return x, y

def find_max(x):
    """ Finds the maximum value for each of the variables

    Args:
        x:   The list of batches as autograd Variables
    
    Returns:
        An autograd Variable the size of a single datapoint
    
    """
    x_max = Variable(torch.ones(len(x), len(x[0][0])))
    for b in range(len(x)):
            x_max[b] = torch.max(x[b], 0)[0].data
    max_v = torch.max(x_max, 0)[0]
    return max_v

def main(folder, save_as, targets, inputs):
    x, y = read_all_files(folder, targets, inputs)
    print("Read %i batches" %len(x))
    metaparams = {'d_inp':len(x[0][0]), 'd_hid':units,
                  'd_out':len(y[0][0]), 'layers':layers,
                  'x_max':find_max(x), 'y_max':find_max(y)}
    x, y = normalize(x, y, metaparams)
    print("Datasets normalized")
    model = train_model(x, y, metaparams)
    print("Trained model for %i epochs" %epochs)
    # Always save as CPU model, cast to CUDA while loading if required
    torch.save(metaparams, save_as + ".meta")
    torch.save(model.float().state_dict(), save_as)
    print("Model saved as",save_as)
    return model

if __name__ == "__main__":
    # targets: accelCmd, brakeCmd, steerCmd, gear
    targets = [0, 1, 2, 8]
    # inputs: angle, speed(X-Z), trackSens(0-18), distToMiddle
    inputs = [3] + list(range(11, 34))
    model = main(sys.argv[1], sys.argv[2], targets, inputs)

data_folder = "C:/Users/Joram/Documents/Studie/torcs-client/train_single/"