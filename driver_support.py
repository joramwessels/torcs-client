import typing as t
import numpy as np

def get_gear(current_rpm, current_gear):
    if current_rpm >= 8500:
        current_gear += 1
    if current_rpm <= 2000:
        current_gear -= 1
    if current_gear <= 0:
        current_gear = 1
    if current_gear >= 6:
        current_gear = 6
    return current_gear

def get_steer(target_pos, actual_pos, angle, epsilon=0.1):
    error = target_pos - actual_pos
    angle = angle + error * epsilon
    steer = angle/180
    return steer

def get_accel(target_speed, actual_speed):
    accel = 0
    if (target_speed - actual_speed) > 0:
        accel = (target_speed - actual_speed)/20
    if target_speed - actual_speed > 20:
        accel = 1
    return accel

def get_break(target_speed, actual_speed):
    brake = 0
    if (target_speed - actual_speed) < 0:
        brake = -(target_speed - actual_speed)/20
    if target_speed - actual_speed < -20:
        brake = 1
    return brake

def map_to_gear(prediction):
	# we represent the reverse gear as index 0
	# then gear=1 is just index 1 of the list, etc.
	index = np.argmax(prediction)
	if index == 0:
		return -1
	else:
		return index

def read_dataset_stear_speed(filename: str) -> t.Iterable[t.Tuple[t.List[float], t.List[float]]]:
	with open(filename, "r") as f:
		next(f) # as the file is a csv, we don't want the first line
		# acc, break, steer,
		# 1.0, 0.0, -1.6378514771737686E-5,
		# speed, pos, angle,
		# -0.0379823,-5.61714E-5,4.30409E-4,
		# distances... len(distances) = 19
		# 5.00028, 5.0778, 5.32202, 5.77526, 6.52976, 7.78305, 10.008, 14.6372, 28.8659, 200.0,
		# 28.7221, 14.6009, 9.99199, 7.7742, 6.52431, 5.77175, 5.31976, 5.07646, 4.99972
		for line in f:
			# we predict based on steer + speed
			# based on pos + angle + distances
			yield ([float(x) for x in line.strip().split(",")[2:4]], [float(x) for x in line.strip().split(",")[4:]])

def read_lliaw_dataset_gear_acc_bre_rpm_spe(filename: str) -> t.Iterable[t.Tuple[t.List[float], t.List[float]]]:
    with open(filename, "r") as f:
    	for line in f:
            #   0     1     2      3      4             5              6       7    8    9
			# accel break steer angle curLapTime distFromStartLine distRaced fuel gear racepos
            # 10    11      12    13     14-22             23          24-49
            # rpm speedx speedy speedz tracksensor1_19 distToMiddle oppSenso1_36
            # y=gear, x=accel, break, rpm, speedx
            line_values = [float(x) for x in line.strip().split(",")[:-1]]
            yield ([line_values[8]], [line_values[0], line_values[1], line_values[10], line_values[11]])

def read_lliaw_dataset_acc_bre_steer_bunch(filename: str) -> t.Iterable[t.Tuple[t.List[float], t.List[float]]]:
    with open(filename, "r") as f:
    	for line in f:
            #   0     1     2      3      4             5              6       7    8    9
			# accel break steer angle curLapTime distFromStartLine distRaced fuel gear racepos
            # 10    11      12    13     14-32             33          34-59
            # rpm speedx speedy speedz tracksensor1_19 distToMiddle oppSenso1_36
            # y=accel, break, stear, x=angle, speed, distance*19, distToMiddle
            line_values = [float(x) for x in line.strip().split(",")[:-1]]
            yield ([line_values[0],line_values[1], line_values[2]], [line_values[3], line_values[11],
                                                                     line_values[14], line_values[15], line_values[16],
                                                                     line_values[17], line_values[18], line_values[19],
                                                                     line_values[20], line_values[21], line_values[22],
                                                                     line_values[23], line_values[24], line_values[25],
                                                                     line_values[26], line_values[27], line_values[28],
                                                                     line_values[29], line_values[30], line_values[31], line_values[32],
                                                                     line_values[33]])


def read_lliaw_dataset_acc_br_stee_ge(filename: str) -> t.Iterable[t.Tuple[t.List[float], t.List[float]]]:
	with open(filename, "r") as f:
	    # accel break steer angle curLapTime distFromStartLine distRaced fuel gear racepos
        # rpm speedx speedy speedz tracksensor1_19 distToMiddle
		for line in f:
			# we predict based on steer + speed
			# based on pos + angle + distances
			yield ([float(x) for x in line.strip().split(",")[0:3] + line.strip().split(",")[8:9]], [float(x) for x in line.strip().split(",")[3:34]])

def read_dataset_acc_break_steer(filename: str) -> t.Iterable[t.Tuple[t.List[float], t.List[float]]]:
	with open(filename, "r") as f:
		next(f) # as the file is a csv, we don't want the first line
		# acc, break, steer,
		# 1.0, 0.0, -1.6378514771737686E-5,
		# speed, pos, angle,
		# -0.0379823,-5.61714E-5,4.30409E-4,
		# distances... len(distances) = 19
		# 5.00028, 5.0778, 5.32202, 5.77526, 6.52976, 7.78305, 10.008, 14.6372, 28.8659, 200.0,
		# 28.7221, 14.6009, 9.99199, 7.7742, 6.52431, 5.77175, 5.31976, 5.07646, 4.99972
		for line in f:
			# we predict based on steer + speed
			# based on pos + angle + distances
			yield ([float(x) for x in line.strip().split(",")[0:3]], [float(x) for x in line.strip().split(",")[3:]])
