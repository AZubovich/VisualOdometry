from imutils import paths
import numpy as np
import cv2
import imutils
import time

KITTI_path = '/home/alexandr/Downloads/Odometry/dataset/sequences/08/image_0'
times_path = '/home/alexandr/Downloads/Odometry/dataset/sequences/08/times.txt'
true_poses_path = '/home/alexandr/Downloads/Odometry/dataset/poses/08.txt'
calibration_path = '/home/alexandr/Downloads/Odometry/dataset/sequences/08/calib.txt'


true_poses = np.zeros((4071,12))
trajectory = np.zeros((600,800,3))

file = open(times_path, 'r')
times = file.readlines()
file.close()

file = open(true_poses_path, 'r')
i = 0
for line in file:
	true_poses[i][:] = line.split(' ')
	i += 1

file.close()

calibration_data = 0
file = open(calibration_path, 'r')
for line in file:
	calibration = line.split(' ')
	break

file.close()

count = 0
sleep_time = 0
start_point = (int(round(true_poses[count][3] + 400)), int(round(true_poses[count][11] + 100)))
end_point = (int(round(true_poses[count][3] + 400)), int(round(true_poses[count][11] + 100)))


for imagePath in sorted(paths.list_images(KITTI_path)):
	img = cv2.imread(imagePath)

	if count != 0:
		sleep_time = float(times[count]) - float(times[count - 1])

	time.sleep(sleep_time)
	cv2.imshow("KITTI dataset", img)

	cv2.putText(trajectory, "True position" ,(40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255), 1)
	cv2.putText(trajectory, "Estimated position", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0), 1)

	if count !=0:
		start_point = (int(round(true_poses[count - 1][3] + 400)), int(round(true_poses[count -1 ][11] + 100)))
		end_point = (int(round(true_poses[count][3] + 400)), int(round(true_poses[count][11] + 100)))

	trajectory = cv2.line(trajectory, start_point, end_point, (0, 0, 255), 1)
	cv2.imshow("trajectory", trajectory)

	print(f'True x : {true_poses[count][3]}, y : {true_poses[count][7]}, z: {true_poses[count][11]}')

	if cv2.waitKey(1) & 0xFF == ord('q'):
		print('Its over, buy')
		break

	count += 1

cv2.destroyAllWindows()
	