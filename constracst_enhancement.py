import numpy as np
import pywt
from math import log10
import matplotlib.pyplot as plt
import scipy.io
import cv2

im = scipy.io.loadmat('Pepsi.mat')['pepsi']
# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl1 = clahe.apply(im)

fig1, axarr1 = plt.subplots(1, 2)
axarr1[0,].imshow(im, cmap='gray')
axarr1[0].set_title('Original')
axarr1[1].imshow(cl1, cmap='gray')
axarr1[1].set_title('New')

fig2, axarr2 = plt.subplots(1, 2)
axarr2[0,].hist(im.flatten(),64,[0,256],ec='black',alpha=0.5)
axarr2[0].set_title('Original')
axarr2[1].hist(cl1.flatten(),64,[0,256],ec='black',alpha=0.5)
axarr2[1].set_title('New')
plt.show()

