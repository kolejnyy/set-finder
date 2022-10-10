import cv2
import numpy as np
from os import listdir, mkdir, makedirs
from random import randint
from tqdm import tqdm

def rotate_image(image, angle):
	image_center = tuple(np.array(image.shape[1::-1]) / 2)
	rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
	result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
	return result

ann_dict = {
    "red":      0,
    "blue":     1,
    "green":    2,
    "oval":     3,
    "romb":     4,
    "wave":     5,
    "clean":    6,
    "strip":    7,
    "full":     8,
    "1":        9,
    "2":        10,
    "3":        11,
    }


def generate_batch(batch_size=64, save=False):

    bcg_path = 'dataset/backgrounds/'
    img_path = 'dataset/clean_cards/'

    backs = [bcg_path+x for x in listdir(bcg_path)]
    cards = [img_path+x for x in listdir(img_path)]

    im_size = 1024

    batch = []

    for i in range(batch_size):
        back_id = randint(0,len(backs)-1)
        card_id = randint(0,len(cards)-1)
        card_size = randint(500,900)
        
        bckg = cv2.imread(backs[back_id])
        bckg = cv2.resize(bckg,(im_size,im_size))

        card = cv2.imread(cards[card_id])
        ratio = card.shape[0]/card.shape[1]
        card = cv2.resize(card, (card_size, round(card_size*ratio)))
        card_shape = card.shape
        
        print(cards[card_id])

        cardback = np.zeros(bckg.shape)
        x = randint(50, im_size-card_shape[0]-50)
        y = randint(50, im_size-card_shape[1]-50)
        cardback[x:x+card_shape[0],y:y+card_shape[1]] = card

        noise = np.random.random(bckg.shape)*randint(0,100)

        angle = randint(-50,50)
        if angle > 25:
            angle = 90-angle
        if angle < -25:
            angle = -90-angle
        cardback = rotate_image(cardback, angle).astype(np.uint8)

        mask = (cardback==0)
        np.putmask(cardback, mask, bckg)
        image = cardback+noise
        if save:
            cv2.imwrite('dataset/random/img_{}.png'.format(i), image)
    
        ann = np.zeros(12)
        for param in cards[card_id].split('/')[-1].replace('.jpg', '').split('_'):
            ann[ann_dict[param]]=1

        print("img_{}.png: {}".format(i,ann))
        batch.append((image, ann))
    
    return batch