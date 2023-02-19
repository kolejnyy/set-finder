from time import time
import cv2
import numpy as np
from os import listdir, mkdir, makedirs
from random import randint
from tqdm import tqdm
from models.recognizer import Recognizer
from generate_data import generate_batch
import torch

recog = Recognizer(256)

recog.load_state_dict(torch.load('weights/recognizer_2899.pth'))

optimizer = torch.optim.Adam(recog.parameters(), lr=0.00005, weight_decay=1e-5)
criterion = torch.nn.CrossEntropyLoss()

epochs = 3000
save_freq = 50

for i in (range(epochs)):
    s_t = time()
    images, ann = generate_batch(32, save=True, im_size=256)
    output = recog(images/255)

    # Evaluate loss
    color_loss = criterion(output[:,0], ann[:,0])
    shape_loss = criterion(output[:,1], ann[:,1])
    fill_loss = criterion(output[:,2], ann[:,2])
    number_loss = criterion(output[:,3], ann[:,3])
    
    loss = color_loss + shape_loss + fill_loss + number_loss

    # Training step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Epoch {}/{}:   loss = {:2.4f}   shape_loss = {:2.4f}    batch_time = {:2.3f}s".format(i+1,epochs,loss.item(),shape_loss.item(), time()-s_t))

    if i%save_freq == save_freq-1:
        torch.save(recog.state_dict(), 'weights/recognizer_{}.pth'.format(i))