import cv2
import numpy as np


class MonoVisualOdometry:

	def __init__(self, calibration_data, true_poses):
		self.focal_length = calibration_data[1]
		self.principal_point = (calibration_data[3], calibration_data[7])
		self.true_poses = true_poses
		self.old_image = None
		self.new_image = None
		self.key_points = None
		self.fast = cv2.FastFeatureDetector_create(threshold=20)
		self.orb = cv2.ORB_create()
		self.lk_params = dict( winSize  = (15,15), maxLevel = 2, 
			criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

	def initial_image(self,img):
		self.new_image = img
		self.mask = np.zeros_like(img)
		self.color = np.random.randint(0,255,(8000,3))
		self.FAST_detector(img)

	def perform(self, img):
		self.old_image = self.new_image
		self.new_image = img
		self.LKT_optical_flow()

	def FAST_detector(self,img):
		kp = self.fast.detect(img,None)
		self.key_points = np.array([x.pt for x in kp], dtype=np.float32).reshape(-1, 1, 2)

	def ORB_detector(self, img):
		kp, des = self.orb.detectAndCompute(img,None)
		self.key_points = np.array([x.pt for x in kp], dtype=np.float32).reshape(-1, 1, 2)

	def LKT_optical_flow(self):
		p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_image, self.new_image, self.key_points, None, **self.lk_params)

		good_new = p1[st==1]
		good_old = self.key_points[st==1]

		for i,(new,old) in enumerate(zip(good_new, good_old)):
			a,b = new.ravel()
			c,d = old.ravel()
			self.mask = cv2.line(self.mask, (a,b),(c,d), self.color[i].tolist(), 2)
			self.new_image = cv2.circle(self.new_image,(a,b),5, self.color[i].tolist(),-1)
    	
		img = cv2.add(self.new_image, self.mask)
		cv2.imshow('LKT optical flow',img)
		self.key_points = good_new.reshape(-1,1,2)
