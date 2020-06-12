import numpy as np

def time_preprocessing(time_path):
	file = open(time_path, 'r')
	times = file.readlines()
	file.close()
	return times

def poses_preprocessing(poses_path):
	true_poses = np.zeros((4071,12))
	file = open(poses_path, 'r')
	i = 0

	for line in file:
		true_poses[i][:] = line.split(' ')
		i += 1

	file.close()
	return true_poses

def calibration_preprocessing(calibration_path):
	calibration_data = 0
	file = open(calibration_path, 'r')
	for line in file:
		calibration_data = line.split(' ')
		break

	file.close()
	return calibration_data
