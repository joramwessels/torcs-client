# Computational Intelligence project team 33


# Authors
Haukur Páll Jónsson, ADD YOURSELVES


# Starting it up

  torcs
  # race-quick-race
  ./start.sh

or

  /home/haukur/ci/torcs-server/torcs_tournament.py /home/haukur/ci/torcs-server/quickrace.yml


# Torcs-Client

See pdf:
http://www.berniw.org/aboutme/publications/torcs.pdf

This is a copy of the reimplementation in Python 3 by @moltob of the original SCRC TORCS client pySrcrcClient from @lanquarden. It is used to teach ideas of computational intelligence. The file `my_driver.py` contains a shell to start writing your own driver.

## Things to do
- Implement data reader which parses driverlogs so that we can use them directly for learning
- Create a simple setup/bot to generate data.
- Expand data generator to have multiple competitors
- Implement speed/angle NN predictor (almost done)
- Add an evolutionary approach (NEAT might be a simple answer)
- Implement safety mechanism, when we go off track, stop or turn around. Return us to the middle of the track.

### Neural network approach
We use distance sensors in around car and angle on road to predict speed and steering.
This provides a good baseline for simply driving on the track.

## `Client`

* top level class
* handles _all_ aspects of networking (connection management, encoding)
* decodes class `State` from message from server, `state = self.decode(msg)`
* encodes class `Command` for message to server, `msg = self.encode(command)`
* internal state connection properties only and driver instance
* use `Client(driver=your_driver, <other options>)` to use your own driver

## `Driver`

* encapsulates driving logic only
* main entry point: `drive(state: State) -> Command`

## `State`

* represents the incoming car state

## `Command`

* holds the outgoing driving command
