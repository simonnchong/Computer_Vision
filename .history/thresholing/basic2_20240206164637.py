import cv2
import matplotlib.pyplot as plt

img = cv2.imread("../DATA/crossword.jpg", 0)
plt.imshow(img, cmap="gray")

def show_pic(img):
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap="gray")

ret, output = sc2.threshold(img, 127,255,cv2.THRESH_BINARY)
show_pic(output)

plt.show()