from imutils import paths
from preprocessing import time_preprocessing, poses_preprocessing, calibration_preprocessing
from mono_visual_odometry import MonoVisualOdometry
import numpy as np
import cv2
import imutils
import time

KITTI_path = '/home/alexandr/Downloads/Odometry/dataset/sequences/08/image_0'
times_path = '/home/alexandr/Downloads/Odometry/dataset/sequences/08/times.txt'
true_poses_path = '/home/alexandr/Downloads/Odometry/dataset/poses/08.txt'
calibration_path = '/home/alexandr/Downloads/Odometry/dataset/sequences/08/calib.txt'

trajectory = np.zeros((800,800,3))

times = time_preprocessing(times_path)
true_poses = poses_preprocessing(true_poses_path)
calibration_data = calibration_preprocessing(calibration_path)


vo = MonoVisualOdometry(calibration_data, true_poses)
count = 0
sleep_time = 0
start_point = (int(round(true_poses[count][3] + 400)), int(round(true_poses[count][11] + 100)))
end_point = (int(round(true_poses[count][3] + 400)), int(round(true_poses[count][11] + 100)))

est_start_point = start_point
est_end_point = end_point


for imagePath in sorted(paths.list_images(KITTI_path)):
	img = cv2.imread(imagePath)

	if count != 0:
		sleep_time = float(times[count]) - float(times[count - 1])
		start_point = (int(round(true_poses[count - 1][3] + 400)), int(round(true_poses[count -1 ][11] + 100)))
		end_point = (int(round(true_poses[count][3] + 400)), int(round(true_poses[count][11] + 100)))

	#time.sleep(sleep_time)

	cv2.imshow("KITTI dataset", img)

	cv2.putText(trajectory, "True position" ,(40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255), 1)
	cv2.putText(trajectory, "Estimated position", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 1)

	if count == 0:
		vo.initial_image(img, count)
	else:
		vo.perform(img, count)
		est_coord = vo.estimate_coordinates()
		#print(vo.MSE_error(est_coord))
		est_end_point = (int(round(est_coord[0][0] + 400)), int(round(est_coord[2][0] + 100)))


	trajectory = cv2.line(trajectory, start_point, end_point, (0, 0, 255), 1)
	trajectory = cv2.line(trajectory, est_start_point, est_end_point, (0, 255, 0), 1)
	
	cv2.imshow("trajectory", trajectory)

	#print(f'True x : {true_poses[count][3]}, y : {true_poses[count][7]}, z: {true_poses[count][11]}')

	if cv2.waitKey(1) & 0xFF == ord('q'):
		print('Its over, buy')
		break

	count += 1
	est_start_point = est_end_point

cv2.imwrite('trajectory.png', trajectory)
cv2.destroyAllWindows()
	