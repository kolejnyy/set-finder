import cv2
import torch
import numpy as np
from random import randint
from models.detector import Detector
from models.recognizer import Recognizer


class VideoAgent():

	def __init__(self, cam_port, res_W = 1920, res_H = 1080,
	      		detector = Detector(180, save=False), 
				recognizer = Recognizer(256),
				rocog_loadpath = 'weights/finetuned.pth') -> None:
		
		self.cam_port = cam_port
		self.res_W = res_W
		self.res_H = res_H
		self.detector = detector
		self.recognizer = recognizer
		self.recognizer.load_state_dict(torch.load(rocog_loadpath))

		self.mask_det = np.zeros((res_H, res_W, 3), dtype=np.uint8)
		self.mask_solve = np.zeros((res_H, res_W, 3), dtype=np.uint8)

		self.decode_dict = [["red", "blue", "green"],
		      				["oval", "romb", "wave"],
							["clean", "strip", "full"],
							["1", "2", "3"]]

		self.cap = None
		self.det = False

	def open(self):
		self.cap = cv2.VideoCapture(self.cam_port)
		self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.res_W)
		self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.res_H)

		# Check if the webcam is opened correctly
		if not self.cap.isOpened():
			raise IOError("Cannot open webcam")

	def close(self):
		self.cap.release()
		cv2.destroyAllWindows()
	
	def prepare_cards(self, cards):
		input = []
		sz = self.recognizer.im_size
		for card in cards:
			recog_img = np.zeros((sz,sz,3), dtype=np.uint8)+169
			try:
				card = cv2.resize(card, (round(4*sz/5),round(4*sz/5*card.shape[0]/card.shape[1])))
				recog_img[round(2*sz/10):card.shape[0]+round(2*sz/10),round(sz/10):round(sz/10)+card.shape[1]] = card
				recog_img = torch.from_numpy(recog_img).permute(2,0,1).float().unsqueeze(0)
				recog_img = recog_img/255
				input.append(recog_img)
			except Exception as e:
				print(card, e)
		return torch.cat(input, dim=0)
	

	def iterate_sets(self, annotations):
		result = []
		for i in range(annotations.shape[0]):
			for j in range(i+1, annotations.shape[0]):
				for k in range(j+1, annotations.shape[0]):
					ann1 = annotations[i]
					ann2 = annotations[j]
					ann3 = annotations[k]
					if np.sum((ann1+ann2+ann3)%3)==0:
						result.append([i,j,k])
		return result
	
	def recognize(self, cards, contours):
		print('len(cards):', len(cards))
		if len(cards) == 0:
			return
		input = self.prepare_cards(cards)
		preds = self.recognizer(input)
		annotations = torch.argmax(preds, dim=2).detach().cpu().numpy()

		for i in range(annotations.shape[0]):
			print("Card #{}: {}".format(i+1, " ".join([self.decode_dict[j][annotations[i,j]] for j in range(4)])))

		sets = self.iterate_sets(annotations)

		print("Found {} sets".format(len(sets)))
		if len(sets) != 0:
			solve = sets[randint(0, len(sets)-1)]
			self.mask_solve = np.zeros((1024, 1024*self.res_W//self.res_H, 3), dtype=np.uint8)
			for i in range(3):
				self.mask_solve = cv2.drawContours(self.mask_solve, contours[solve[i]], -1, (1, 1, 255), 15)
			self.mask_solve = cv2.resize(self.mask_solve, (self.res_W, self.res_H))


	def clear_masks(self):
		self.mask_det[:,:,:] = 0
		self.mask_solve[:,:,:] = 0

	def run(self):

		cards = None
		cnts = None

		while True:
			ret, frame = self.cap.read()
			char = cv2.waitKeyEx(1)

			if char == 27:
				break

			if char == ord('q'):
				self.clear_masks()
				self.det = False
			
			if char == ord('s'):
				cv2.imwrite('frame.jpg', frame)
				print('Saved frame')

			if char == ord('l'):
				self.det = False
				print('Loading frame')
				self.mask_det = cv2.imread('frame.jpg')
				self.mask_det = cv2.resize(self.mask_det, (self.res_W, self.res_H))
				print(self.mask_det.shape)

			if char == ord('d'):
				self.det = not self.det
			
			if char == ord('o'):
				self.detector.set_threshold(self.detector.thr+1)

			if char == ord('p'):
				self.detector.set_threshold(self.detector.thr-1)

			if self.det:
				cv2.imwrite('board.jpg', frame)

				cards, cnts, shp = self.mask_det = self.detector.detect('board.jpg')
				self.mask_det = np.zeros((1024, 1024*self.res_W//self.res_H, 3), dtype=np.uint8)
				self.mask_det = cv2.drawContours(self.mask_det, cnts, -1, (255, 255, 1), 5)
				self.mask_det = cv2.resize(self.mask_det, (self.res_W, self.res_H))

			if char == ord('r'):
				self.recognize(cards, cnts)


			total_mask = np.where(self.mask_solve!=0, self.mask_solve, self.mask_det)
			frame = np.where(self.mask_det+self.mask_solve != 0, self.mask_det+self.mask_solve, frame)
			frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
			cv2.imshow('SetFinder', frame)