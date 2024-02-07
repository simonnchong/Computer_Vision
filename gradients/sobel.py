
import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_img(img):  # this function just makes the image being displayed larger
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111,)
    ax.imshow(img,cmap="gray")

img = cv2.imread("../DATA/sudoku.jpg", 0)
# display_img(img)

sobel_x = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
# display_img(sobel_x)

sobel_y = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
# display_img(sobel_y)

blended = cv2.addWeighted(src1=sobel_x,alpha=0.5,src2=sobel_y,beta=0.5,gamma=0)
display_img((blended))

plt.show()