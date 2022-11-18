import cv2
import numpy as np
from os import listdir, mkdir, makedirs
from random import randint
from tqdm import tqdm
import torch
from torch import from_numpy

def rotate_image(image, angle):
	image_center = tuple(np.array(image.shape[1::-1]) / 2)
	rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
	result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
	return result

ann_dict = {
    "red":      0,
    "blue":     1,
    "green":    2,
    "oval":     0,
    "romb":     1,
    "wave":     2,
    "clean":    0,
    "strip":    1,
    "full":     2,
    "1":        0,
    "2":        1,
    "3":        2,
    }


def generate_batch(batch_size=64, save=False, im_size=512, multiorient=False):

    bcg_path = 'dataset/backgrounds/'
    img_path = 'dataset/clean_cards/'

    backs = [bcg_path+x for x in listdir(bcg_path) if x.endswith('.jpg')]
    cards = [img_path+x for x in listdir(img_path) if x.endswith('.jpg')]

    batch_img = []
    batch_ann = []

    for i in range(batch_size):
        back_id = randint(0,len(backs)-1)
        card_id = randint(0,len(cards)-1)
        card_size = randint(round(0.74*im_size),round(0.89*im_size))
        
        bckg = cv2.imread(backs[back_id])
        bckg = cv2.resize(bckg,(im_size,im_size))

        card = cv2.imread(cards[card_id])
        ratio = card.shape[0]/card.shape[1]
        card = cv2.resize(card, (card_size, round(card_size*ratio)))
        card_shape = card.shape

        cardback = np.zeros(bckg.shape)
        border = round(im_size*0.05)
        x = randint(border, im_size-card_shape[0]-border)
        y = randint(border, im_size-card_shape[1]-border)
        cardback[x:x+card_shape[0],y:y+card_shape[1]] = card

        noise = np.random.random(bckg.shape)*randint(0,100)

        angle = randint(-30,30)
        if angle > 15 and multiorient:
            angle = 90-angle
        if angle < -15 and multiorient:
            angle = -90-angle
        cardback = rotate_image(cardback, angle).astype(np.uint8)

        mask = (cardback==0)
        np.putmask(cardback, mask, bckg)
        image = cardback+noise
        if save:
            cv2.imwrite('dataset/random/img_{}.png'.format(i), image)
    
        ann = []
        for param in cards[card_id].split('/')[-1].replace('.jpg', '').split('_'):
            ann.append(ann_dict[param])
        ann = np.array(ann)

        image = np.transpose(image,(2,0,1))
        batch_img.append(from_numpy(image).type('torch.FloatTensor'))
        batch_ann.append(from_numpy(ann).type('torch.LongTensor'))
    
    batch_img = torch.stack(batch_img)
    batch_ann = torch.stack(batch_ann)
    
    return batch_img, batch_ann