import numpy as np
import torch
from torch import functional as F
from torch import nn, optim
import cv2

class Detector:

	def __init__(self, thr = 180, save = False) -> None:
		self.thr = thr
		self.save_mode = save

	def extract_contours(self, image):
		# Binarize image
		_, im = cv2.threshold(image, self.thr, 255, cv2.THRESH_BINARY)
		if self.save_mode:
			cv2.imwrite('debug/binarized.jpg', im)
		
		# Find contours
		im = im.astype(np.uint8)
		contours, _ = cv2.findContours(im, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		contours = [x for x in contours if cv2.contourArea(x)>1500]
		if self.save_mode:
			new = cv2.drawContours(np.zeros(im.shape), contours, -1, (255,255,255), 1)
			cv2.imwrite('debug/contours.jpg', new)
		
		return contours
	
	def get_ranges(self, contours):
		ranges = []
		i=0
		for ctr in contours:
			x,y,w,h = cv2.boundingRect(ctr)
			ranges.append((x,y,w+x,h+y,i))
			i+=1
		return ranges
	
	def prepare_image(self, path):
		im = cv2.imread(path,0)
		shp = im.shape
		ratio = shp[0]/shp[1]
		im = cv2.resize(im, (round(1024/ratio),1024))
		return im
	
	
	def draw_card_contours(self, path, card_contours, shape):
		card_cnt = cv2.resize(cv2.imread(path), shape)
		card_cnt = cv2.drawContours(card_cnt, card_contours, -1, (255,0,255), 3)
		if self.save_mode:
			cv2.imwrite('debug/card_contours.jpg', card_cnt)

	def insider(self, ranges, i):
		x,y,w,h,_ = ranges[i]
		for j in range(len(ranges)):
			if i==j:
				continue
			x1,y1,w1,h1,_ = ranges[j]
			if x1<x and y1<y and w1>w and h1>h:
				return True
		return False


	def crop_card(self, image, center, theta, width, height):

		shape = (image.shape[1], image.shape[0]) 
		matrix = cv2.getRotationMatrix2D(center=center, angle=theta, scale=1)
		image = cv2.warpAffine(src=image, M=matrix, dsize=shape)

		x = round(center[0] - width/2)
		y = round(center[1] - height/2)

		return image[y:round(y+height), x:round(x+width)]

	def get_card(self, im, contour):
		rect = cv2.minAreaRect(contour)
		cropped = self.crop_card(im, rect[0], rect[2], rect[1][0], rect[1][1])
		if cropped.shape[0]>cropped.shape[1]:
			cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)
		return cropped

	def detect(self, path):
		im = self.prepare_image(path)
		contours = self.extract_contours(im)
		ranges = self.get_ranges(contours)
		valid_contours = [x[4] for i,x in enumerate(ranges) if not self.insider(ranges, i)]
		card_contours = [contours[x] for x in valid_contours]

		self.draw_card_contours(path, card_contours, im.transpose().shape)

		clr_im = cv2.resize(cv2.imread(path), im.transpose().shape) 
		cards = [self.get_card(clr_im, x) for x in card_contours]
		return cards, card_contours, im.transpose().shape

	
	def set_threshold(self, thr):
		self.thr = thr