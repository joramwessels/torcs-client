# Computational Intelligence project team 32

# Authors
Haukur Páll Jónsson, ADD YOURSELVES

# Things to do
- Test actual setup for turn in
- This includes checking if we need to worry about ports
- Implement data reader which parses driverlogs so that we can use them directly for learning
- Create a simple setup/bot to generate data.
- Expand data generator to have multiple competitors
- Implement speed/angle FFNN predictor (almost done)
- Implement simple (acc/break/steer) predictor FFNN
- Add an evolutionary approach (NEAT might be a simple answer)
- Implement safety mechanism, when we go off track, stop or turn around. Return us to the middle of the track.
- Implement sec2sec NN

# Starting it up
Start the server by calling

  torcs

And then start the client (in a different process)

  ./start.sh

To start a race without GUI. See Configuration [Configuration]

  ./torcs_tournament.py quickrace.yml


# Torcs-Client

- TORCS manuals
* https://arxiv.org/pdf/1304.1672.pdf
* http://www.berniw.org/aboutme/publications/torcs.pdf

- Papers
* http://ieeexplore.ieee.org/document/5286480/?reload=true
* http://ieeexplore.ieee.org/abstract/document/7848001/
* http://ieeexplore.ieee.org/abstract/document/7317916/
* http://julian.togelius.com/Togelius2006Making.pdf

- Useful links:
* Server https://github.com/mpvharmelen/torcs-server
* Client https://github.com/mpvharmelen/torcs-client
* Blog http://www.xed.ch/help/torcs.html

## Configuration
ls /usr/local/share/games/torcs/drivers/
berniw   berniw3  damned  inferno   lliaw     scr_server  tita
berniw2  bt       human   inferno2  olethros  sparkle

# Learning

## Neural network approach
We use distance sensors in around car and angle on road to predict speed and steering.
This provides a good baseline for simply driving on the track.
