import typing as t

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
