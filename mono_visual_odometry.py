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
		self.R = None #np.zeros(shape=(3, 3))
		self.t = None #np.zeros(shape=(3, 3))
		self.fast = cv2.FastFeatureDetector_create(threshold=20)
		self.orb = cv2.ORB_create()
		self.lk_params = dict( winSize  = (21,21), maxLevel = 3, 
			criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

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
		self.key_points = np.array([x.pt for x in kp], dtype=np.float32)

	def ORB_detector(self, img):
		kp, des = self.orb.detectAndCompute(img,None)
		self.key_points = np.array([x.pt for x in kp], dtype=np.float32).reshape(-1, 1, 2)

	def Shi_Tomasi_detector(self,img):
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
		self.key_points = np.array([x for x in corners], dtype=np.float32).reshape(-1, 1, 2)

	def LKT_optical_flow(self):
		if self.key_points.shape[0] < 2000:
			self.FAST_detector(self.old_image)

		p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_image, self.new_image, self.key_points, None, **self.lk_params)
		#p2, st, err = cv2.calcOpticalFlowPyrLK(self.new_image, self.old_image, p1, None, **self.lk_params)
		st = st.reshape(st.shape[0])
		good_new = p1[st==1]
		good_old = self.key_points[st==1]
		#good_new = p2[st==1]
		#good_old = p1[st==1]
		self.estimate_pose(good_old, good_new)

	def estimate_pose(self, good_old, good_new):

		if self.count < 2:
			E, mask = cv2.findEssentialMat(good_new, good_old, focal=self.focal_length,
                         pp=self.principal_point, 
                         method=cv2.RANSAC, 
                         prob=0.999, 
                         threshold=0.1)

			#self.R, R2, self.t = cv2.decomposeEssentialMat(E)
			_, self.R, self.t, mask = cv2.recoverPose(E, good_new, good_old, focal=self.focal_length, pp=self.principal_point)

		else:
			E, mask = cv2.findEssentialMat(good_new, good_old, focal=self.focal_length, pp=self.principal_point, method=cv2.RANSAC, prob=0.999, threshold=0.1)
			#R1, R2, t = cv2.decomposeEssentialMat(E)
			_, R, t, _ = cv2.recoverPose(E, good_new, good_old, focal=self.focal_length, pp=self.principal_point)

			scale = self.compute_scale()
			if scale > 0.1:
				self.t = self.t + scale*self.R.dot(t)
				self.R = self.R.dot(R) # or R1

		self.key_points = good_new

	def estimate_coordinates(self):
		return self.t

	def MSE_error(self, estimated):
		x = self.true_poses[self.count][3]
		y = self.true_poses[self.count][7]
		z = self.true_poses[self.count][11]

		truth = np.array([[x], [y], [z]])

		return np.linalg.norm(truth - estimated)

	def compute_scale(self):
		x_prev = self.true_poses[self.count-1][3]
		y_prev = self.true_poses[self.count-1][7]
		z_prev = self.true_poses[self.count-1][11]

		x = self.true_poses[self.count][3]
		y = self.true_poses[self.count][7]
		z = self.true_poses[self.count][11]

		return np.sqrt((x - x_prev) * (x - x_prev) + (y - y_prev) * (y - y_prev) + (z - z_prev) * (z - z_prev))

