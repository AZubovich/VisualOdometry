import cv2

class MonoVisualOdometry:

	def __init__(self, calibration_data, true_poses):
		self.focal_length = calibration_data[1]
		self.principal_point = (calibration_data[3], calibration_data[7])
		self.true_poses = true_poses
		self.old_image = None
		self.new_image = None

	def initial_image(self,img):
		self.new_image = img
		self.FAST_detector(img)

	def perform(self, img):
		self.old_image = self.new_image
		self.new_image = img
		self.FAST_detector(img)

	def FAST_detector(self,img):
		fast = cv2.FastFeatureDetector_create(threshold=20)
		kp = fast.detect(img,None)
		img1 = cv2.drawKeypoints(img, kp, None, color=(0,255,0))
		cv2.imshow('FAST algorithm', img1)

	def ORB_detector(self, img):
		orb = cv2.ORB_create()
		kp, des = orb.detectAndCompute(img,None)
		img1 = cv2.drawKeypoints(img, kp, None, color=(0,255,0))
		cv2.imshow('ORB algorithm', img1)

