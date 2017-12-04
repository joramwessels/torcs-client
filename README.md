# Computational Intelligence project team 32

# Authors
Haukur Páll Jónsson, Joram Wessels, Yi-ting Lin

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

## Data generation
The best bots are: lliaw, inferno, olethros, tita

Total bots:
berniw   berniw3  damned  inferno   lliaw     scr_server  tita
berniw2  bt       human   inferno2  olethros  sparkle

To generate data call (it assumes linux and lliaw.cpp compiled with TORCS)

  python3.6 data_generator.py --drivers lliaw --tracks all --length 0 --laps 1 --file quickrace.xml

## Data
-  0     1     2      3      4             5              6       7    8    9
- accel break steer angle curLapTime distFromStartLine distRaced fuel gear racepos
- 10    11      12    13     14-33             34          35-70
- rpm speedx speedy speedz tracksensor1_19 distToMiddle oppSenso1_36

# Learning

## Neural network approach
We use distance sensors in around car and angle on road to predict speed and steering.
This provides a good baseline for simply driving on the track.
