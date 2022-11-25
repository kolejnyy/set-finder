from time import time, sleep
import cv2
import numpy as np
from os import listdir, mkdir, makedirs, system
from random import randint
from tqdm import tqdm
from models.recognizer import Recognizer
from models.detector import Detector
from generate_data import generate_batch
import torch
from os import system, listdir
from os.path import isfile, join
from test_model import test_model, draw_accuracy_graphs
import matplotlib.pyplot as plt



class SetFinder():

	def __init__(self, 	detector : Detector,
						recognizer : Recognizer,
						recog_resolution : int = 256,
						save_mode : bool = False):
		
		self.detector = detector
		self.recognizer = recognizer
		self.recog_resolution = recog_resolution
		self.save_mode = save_mode
		self.decode_dict = [["red", "blue", "green"],
							["oval", "romb", "wave"],
							["clean", "strip", "full"],
							["1", "2", "3"]]
		
		self.recognizer.eval()
		
	def prepare_input(self, cards):
		input = []
		sz = self.recog_resolution
		for card in cards:
			recog_img = np.zeros((256,256,3), dtype=np.uint8)+169

			card = cv2.resize(card, (round(4*sz/5),round(4*sz/5*card.shape[0]/card.shape[1])))
			recog_img[round(2*sz/10):card.shape[0]+round(2*sz/10),round(sz/10):round(sz/10)+card.shape[1]] = card

			if self.save_mode:
				cv2.imwrite('debug/recog_img.jpg', recog_img)
			
			recog_img = torch.from_numpy(recog_img).permute(2,0,1).float().unsqueeze(0)
			recog_img = recog_img/255
			input.append(recog_img)

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
						result.append([ann1,ann2,ann3])
		return result

	def find_sets(self, path):
		cards, contours, shp = (self.detector.detect(path))
		if len(cards)==0:
			raise Exception("No cards found")
		input = self.prepare_input(cards)
		output = torch.argmax(self.recognizer(input), dim=2).numpy()

		print("Detected cards:")
		for i in range(output.shape[0]):
			print("Card #{}: {}".format(i+1, " ".join([self.decode_dict[j][output[i,j]] for j in range(4)])))
		print('\n\n\n')
		sets = self.iterate_sets(output)
		print("There are {} sets in this photo".format(len(sets)))
		print('-'*30)
		for set in sets:
			for card in set:
				print(' '.join([self.decode_dict[i][j] for i,j in enumerate(card)]))
			print('-'*20)