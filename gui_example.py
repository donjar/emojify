import numpy as np
import cv2
import matplotlib as mpl
from matplotlib import pyplot as plt
import time
import matplotlib.image as image

im = image.imread('emojis/dizzy_face.png')
im = image.imread('emojis/dizzy_face.png')

plt.figimage(im, 40, 80)

img1 = cv2.imread("image-1.png", 0)
img2 = cv2.imread("image-2.png", 0)
img = img1

plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])

# plt.show()

# attempt to print out emoji svg

toggle = False

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if toggle:
        img = img1
    else:
        img = img2
    toggle = not toggle
    plt.imshow(frame, cmap = 'gray', interpolation = 'bicubic')
    # plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.pause(0.1)

