import cv2
import matplotlib.pyplot as plt

img = cv2.imread("../DATA/crossword.jpg", 0)
plt.imshow(img, cmap="gray")

def show_pic(img):
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap="gray")



plt.show()