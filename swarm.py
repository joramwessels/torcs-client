# file:         swarm.py
# author:       Joram Wessels
# date:         05-12-2017
# depedencies:  numpy
# description:
#       Handles the swarm technology behind the autonomous TORCS driver.
# usage:
#       Assuming the variables
#           *distTraveled*, *pos*, *spd*, *crashed*, and *contact*,
# 
#       if distTraveled % trail.pos_int == 0:
#           max_speed = trail.check_in(pos, spd, crashed, contact)
#           crashed, contact = False, False
# 

from collections import defaultdict
import numpy as np

SEP = ','
NAME = '.feromones'

# - TODO unit test: to_index, to_speed, back_prop, report_result, check_in
# - TODO write: breakable_speed
# - TODO if track is known, or if feromone trail looks like a known track, switch to known trail
# - NOTE if they drive close behind each other, they'll always explore the same max_speeds
# - NOTE initializing a FeromoneTrail will not read the previous .feromone entries

class FeromoneTrail:

    def __init__(self, pos_int, spd_int, spd0, spd_n, expl_int, glb_max,
                 track_id=None):
        """ FeromoneTrail keeps track of-, and syncs the feromone trail

        A FeromoneTrail contains a table of known feromones and syncs to
        a .feromone file in storage to communicate with the swarm. The
        initialization requires a position interval in game cycles at
        which to check and update the feromone trail, a grid of possible
        max_speed values to explore, an exploration interval to increase
        the max_speed with when no negative experiences are known, and
        a global maximum speed to default to when there are no positive
        experiences. The resulting max_speeds can be lower than the
        global maximum when this speed resulted in a negative experience.

        Args:
            pos_int:    The interval at which to check feromones (int)
            spd_int:    The interval between speed boxes (int)
            spd0:       The first speed box (int)
            spd_n:      The amount of speed boxes (int)
            expl_int:   The jump interval when no - values are known (int)
            glb_max:    The global max speed that ensures a finish (int)
            track_id:   The name of the race track if known

        """
        self.pos_int = pos_int
        self.spd_int = spd_int
        self.spd0 = spd0
        self.spd_n = spd_n
        self.spd_max = spd0 + spd_n * spd_int
        self.expl_int = expl_int
        self.glb_max = glb_max
        self.prev_pos = 0
        self.prev_spd = 0
        self.filename = NAME + '_' + track_id if track_id else NAME
        self.table = defaultdict(lambda: np.zeros(spd_n))
        self.leave_feromone(0, 0, 0)
    
    def ___str___(self):
        """ Casts the feromone trail table to a string representation """
        return self.__repr__()

    def __repr__(self):
        """ Casts the feromone trail table to a string representation """
        i = 0
        speeds = [str(self.to_speed(i)) for i in range(self.spd_n)]
        string = "\t " + ' '.join(speeds) + '\n'
        while str(i) in self.table:
            string += str(i) + ':\t' + str(self.table[str(i)]) + '\n'
            i += self.pos_int
        return string

    def to_index(self, spd):
        """ Converts absolute speed to table index """
        return (spd - self.spd0) // self.spd_int

    def to_speed(self, ind):
        """ Converts table index to absolute speed """
        return self.spd0 + ind * self.spd_int

    def write_feromone(self, pos, speed, val):
        """ Writes a new feromone to the .feromone file

        Args:
            pos:    The position on the track, CurLapTime (int)
            speed:  The speed that has been tested (int)
            val:    The result of the test (-1, 0, 1)

        """
        file = open(self.filename, 'a')
        file.write('\n' + SEP.join([str(pos), str(speed), str(val)]))
        file.close()

    def read_feromone(self):
        """ Reads the last feromones and updates it if they're new

        Returns:
            List of [pos, speed, val] lists if there are any
        
        """
        file = open(self.filename, 'r')
        contents = file.readlines()
        file.close()
        i = 1
        changes = []
        changed = True
        while changed:
            if contents[-i].strip() == '': continue
            feromone = [int(s) for s in contents[-i].strip().split(SEP)]
            if feromone == self.last_change: break
            changes.append(feromone)
            i += 1
        if changes: self.last_change = changes[0]
        return changes

    def update_table(self, pos, spd, val):
        """ Updates a newly received feromone in the table

        Args:
            pos:    The position on the track, CurLapTime (int)
            spd:    The speed that has been tested (int)
            val:    The result of the test (-1, 0, 1)

        """
        index = self.to_index(spd)
        self.table[str(pos)][index] = val

    def next_experiment(self, pos):
        """ Checks the table for the next best max speed experiment

        Returns the ideal next max speed to try out, regardless
        of the current speed of the car.

        Args:
            pos:    The position on the track, CurLapTime (int)
        
        Returns:
            The next best max speed value (int)
        
        """
        row = self.table[str(pos)]
        i1 = find_first(row, 1, rev=True)
        i2 = find_first(row, -1)
        i_glb_max = self.to_index(self.glb_max)

        # if there are no occurences of +
        if i1 == -1:
            if row[i_glb_max] == -1:
                i1 = i2 - 1                      # last 0 before first -
            else:
                i1 = i_glb_max                   # resort to global max
        
        # exploring, value in between known values, or safe value
        if i2 == -1:
            spd = min(self.spd_max, self.to_speed(i1) + self.expl_int)
            index = self.to_index(spd)
        else:
            index = i1 + (i2 - i1) // 2
        return index * self.spd_int + self.spd0

    def leave_feromone(self, pos, spd, val):
        """ Updates the table and writes the new feromone to the file

        If an off-grid pos value is passed,
        it defaults to the last on-grid value

        Args:
            pos:    The position on the track, CurLapTime (int)
            spd:    The speed that has been tested (int)
            val:    The result of the test (-1, 0, 1)

        """
        self.last_change = [pos, spd, val]
        self.update_table(pos, spd, val)
        self.write_feromone(pos, spd, val)
    
    def back_prop(self, pos, max_spd):
        """ Updates previous frames to anticipate this failed *max_spd*

        Args:
            pos:        The position on the track, CurLapTime (int)
            max_spd:    The max speed that has failed (int)
        
        """
        while max_spd < self.spd_max:
            first_minus = find_first(self.table[str(pos)], -1)
            if self.to_index(max_spd) >= first_minus:
                break
            self.leave_feromone(pos, max_spd, -1)
            max_spd = breakable_speed(max_spd, self.pos_int)
            max_spd -= max_spd % self.spd_int
            pos -= self.pos_int

    def get_max_speed(self, pos):
        """ Updates the feromone table and returns the next max speed

        If an off-grid pos value is passed,
        it defaults to the next on-grid value

        Args:
            pos:    The position on the track, CurLapTime (int)
        
        Returns:
            The next best max speed value (int)
        
        """
        if not pos % self.pos_int == 0:
            print("SWARM WARNING: Invalid position:",pos)
            pos += self.pos_int - (pos % self.pos_int)
            print("               Defaulted to",pos)
        change = self.read_feromone()
        while change:
            ppos, speed, val = change.pop()
            self.update_table(ppos, speed, val)
        max_speed = self.next_experiment(pos)
        return max_speed

    def report_result(self, pos, spd, val):
        """ Updates the feromone trail with the new information

        Args:
            pos:    The position on the track, CurLapTime (int)
            spd:    The current speed of the car (float)
            val:    The result of the experiment (-1, 0, 1)

        """
        max_spd = int(spd - (spd % self.spd_int) + self.spd_int)
        spd_i = self.to_index(max_spd)
        if val == -1:
            self.back_prop(pos, max_spd)
        elif not self.table[str(pos)][spd_i] == val:
            self.leave_feromone(pos, max_spd, val)

    def check_in(self, pos, spd, crashed, contact):
        """ Called at the start of ever frame to check/update feromones

        Args:
            pos:        The position on the track, CurLapTime (int)
            spd:        The current speed of the car (float)
            crashed:    Indicates a crash or off-track in last frame (bool)
            contact:    Indicates contact with another car in last frame (bool)
        
        Returns:
            The maximum speed for the next frame according to the swarm
        
        """
        assert type(pos) is int, "SWARM WARNING: type(pos) = "+str(type(pos))
        if not pos % self.pos_int == 0:
            print("SWARM WARNING: Invalid position:",pos)
            pos -= pos % self.pos_int
            print("               Defaulted to",pos)
        assert type(spd) is float, "SWARM WARNING: type(spd) = "+str(type(spd))

        if crashed and not contact:
            self.report_result(self.prev_pos, self.prev_spd, -1)
        elif not crashed and not contact:
            self.report_result(self.prev_pos, self.prev_spd, 1)
        self.prev_pos, self.prev_spd = pos, spd
        max_speed = self.get_max_speed(pos)
        return max_speed

def find_first(array, val, rev=False):
    """ Finds the first (or last) occurence of val in array

    Args:
        array:  The numpy array to evaluate
        val:    The value to find
        rev:    If True, returns the last occurence of val
    
    Returns:
        The index of the first (or last) occurence of val
        in array, or -1 if the value doesn't appear in array
    
    """
    ar = np.array(list(array)) # copies the array
    if rev:
        ar = np.flip(ar, 0)
    i = np.argmax(ar==val)
    if i == 0 and not ar[0] == val:
        return -1
    if rev:
        i = abs(i - len(ar) + 1)
    return i

def breakable_speed(end_speed, trajectory):
    """ Computes the max speed that can break to reach *end_speed*

    Args:
        end_speed:  The speed after maximum desceleration
        trajectory: The distance over which to descelerate
    
    Returns:
        The maximum absolute speed at the beginning of the
        trajectory that allows a desceleration to *end_speed*
    
    """
    return # TODO

"""
import swarm
trail = swarm.FeromoneTrail(
            swarm_pos_int, swarm_spd_int,
            swarm_spd0,    swarm_spd_n,
            swarm_expl_int, global_max_speed)

pos = ...
spd = ...
crashed = ...
contact = ...

if distTraveled % trail.pos_int == 0:
    max_speed = trail.check_in(pos, spd, crashed, contact)
    crashed, contact = False, False
"""