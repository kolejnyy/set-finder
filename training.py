from time import time
import cv2
import numpy as np
from os import listdir, mkdir, makedirs
from random import randint
from tqdm import tqdm
from models.recognizer import Recognizer
from generate_data import generate_batch
import torch

recog = Recognizer(512)


optimizer = torch.optim.Adam(recog.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

epochs = 1000

for i in (range(epochs)):
    s_t = time()
    images, ann = generate_batch(32, save=True)
    output = recog(images)

    # Evaluate loss
    loss = criterion(output[:,0], ann[:,0])
    loss += criterion(output[:,1], ann[:,1])
    loss += criterion(output[:,2], ann[:,2])
    loss += criterion(output[:,3], ann[:,3])
    
    # Training step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Epoch {}/1000:\t loss = {:2.4f}  \t batch_time = {:2.3f}s".format(i+1,loss.item(), time()-s_t))

    if i%50 == 49:
        torch.save(recog.state_dict(), 'weights/recognizer_{}.pth'.format(i))