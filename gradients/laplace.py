
import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_img(img):  # this function just makes the image being displayed larger
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111,)
    ax.imshow(img,cmap="gray")

img = cv2.imread("../DATA/sudoku.jpg", 0)

laplacian = cv2.Laplacian(img,cv2.CV_64F)

display_img(laplacian)


plt.show()