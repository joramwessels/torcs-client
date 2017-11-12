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

def read_dataset(filename):
    with open(filename, "r") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            yield ([w2i[x] for x in words.split(" ")], t2i[tag])

# Read in the data
train = list(read_dataset("data/classes/train.txt"))
w2i = defaultdict(lambda: UNK, w2i)
dev = list(read_dataset("data/classes/test.txt"))
nwords = len(w2i)
ntags = len(t2i)


class SimpleNN(nn.Module):

    def __init__(self, nstates, ncommands):
        super(SimpleNN, self).__init__()
        self.linear = nn.Linear(nstates, ncommands, bias=True)

    def forward(self, inputs):
        embeds = self.linear(inputs)
        logits = torch.sum(embeds, 1) + self.bias
        return logits


model = SimpleNN(nwords, ntags)
print(model)

optimizer = optim.SGD(model.parameters(), lr=0.01)

for ITER in range(100):

    random.shuffle(train)
    train_loss = 0.0
    start = time.time()

    for words, tag in train:

        # forward pass
        lookup_tensor = Variable(torch.LongTensor([words]))
        scores = model(lookup_tensor)
        loss = nn.CrossEntropyLoss()
        target = Variable(torch.LongTensor([tag]))
        output = loss(scores, target)
        train_loss += output.data[0]

        # backward pass
        model.zero_grad()
        output.backward()

        # update weights
        optimizer.step()

    print("iter %r: train loss/sent=%.4f, time=%.2fs" %
          (ITER, train_loss/len(train), time.time()-start))

    # evaluate
    _, _, acc = evaluate(model, dev)
    print("iter %r: test acc=%.4f" % (ITER, acc))
