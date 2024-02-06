import cv2
import matplotlib.pyplot as plt

img = cv2.imread("DATA/rainbow.jpg", 0)

plt.imshow(img, cmap="gray")
cv2.threshold(img, 100, 255, )