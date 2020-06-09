from imutils import paths
import numpy as np
import cv2
import imutils
import time

KITTI_path = '/home/alexandr/Downloads/Odometry/dataset/sequences/01/image_0'
times_path = '/home/alexandr/Downloads/Odometry/dataset/sequences/01/times.txt'

file = open(times_path, 'r')
times = file.readlines()
file.close()

sleep_count = 0
sleep_time = 0

for imagePath in sorted(paths.list_images(KITTI_path)):
	img = cv2.imread(imagePath)

	if sleep_count != 0:
		sleep_time = float(times[sleep_count]) - float(times[sleep_count - 1])

	time.sleep(sleep_time)
	cv2.imshow("KITTI dataset", img)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		print('Its over, buy')
		break

	sleep_count += 1

cv2.destroyAllWindows()
	