import cv2
import time
import numpy as np
import os
from os.path import isfile, join

pathIn= '/home/kaj/Documents/Uni/Exjobb/Dataset/NFC/Konferens/2/'

pathOut = 'video.avi'

fps = 20
frame_array = []
files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

#for sorting the file names properly
files.sort(key = lambda x: x[5:-4])
files.sort()
img = cv2.imread(pathIn+files[0])

# get params for VideoWriter
height, width, layers = img.shape
size = (width,height)

out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

for i in range(len(files)):
    filename=pathIn + files[i]
    #reading each files
    img = cv2.imread(filename)

    time.sleep(0.05)
    out.write(img)

out.release()
