import cv2
import numpy as np


class MonoVisualOdometry:

	def __init__(self, calibration_data, true_poses):
		self.focal_length = float(calibration_data[1])
		self.principal_point = (float(calibration_data[3]), float(calibration_data[7]))
		self.true_poses = true_poses
		self.old_image = None
		self.new_image = None
		self.key_points = None
		self.R = np.zeros(shape=(3, 3))
		self.t = np.zeros(shape=(3, 3))
		self.fast = cv2.FastFeatureDetector_create(threshold=20)
		self.orb = cv2.ORB_create()
		self.lk_params = dict( winSize  = (15,15), maxLevel = 2, 
			criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

	def initial_image(self, img, count):
		self.new_image = img
		self.count = count
		self.FAST_detector(img)

	def perform(self, img, count):
		self.old_image = self.new_image
		self.new_image = img
		self.count = count
		self.LKT_optical_flow()

	def FAST_detector(self,img):
		kp = self.fast.detect(img,None)
		self.key_points = np.array([x.pt for x in kp], dtype=np.float32).reshape(-1, 1, 2)

	def ORB_detector(self, img):
		kp, des = self.orb.detectAndCompute(img,None)
		self.key_points = np.array([x.pt for x in kp], dtype=np.float32).reshape(-1, 1, 2)

	def LKT_optical_flow(self):

		if self.key_points.shape[0] < 1800:
			self.FAST_detector(self.old_image)

		p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_image, self.new_image, self.key_points, None, **self.lk_params)
		good_new = p1[st==1]
		good_old = self.key_points[st==1]
		self.estimate_pose(good_old, good_new)

	def estimate_pose(self, good_old, good_new):

		if self.count < 2:
			E, mask = cv2.findEssentialMat(good_old, good_new, self.focal_length,
                         self.principal_point, 
                         cv2.RANSAC, 
                         0.999, 
                         0.1)

			self.R, R2, self.t = cv2.decomposeEssentialMat(E)

		else:
			E, mask = cv2.findEssentialMat(good_old, good_new, self.focal_length, self.principal_point, cv2.RANSAC, 0.999, 0.1)
			R1, R2, t = cv2.decomposeEssentialMat(E)

			scale = 1
			self.R = self.R.dot(R1)
			self.t = self.t + scale*self.R.dot(t)


		self.key_points = good_new.reshape(-1,1,2)

	def estimate_coordinates(self):

		diag = np.array([[-1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]])

		coord = np.matmul(diag, self.t)
		return coord.flatten()

	def MSE_error(self, estimated):
		x = self.true_poses[self.count][3]
		y = self.true_poses[self.count][7]
		z = self.true_poses[self.count][11]

		truth = np.array([[x], [y], [z]])

		return np.linalg.norm(truth - estimated)

