import cv2
import matplotlib.pyplot as plt

img = cv2.imread("DATA/rainbow.jpg", 0)

plt.imshow(img, cmap="gray")

ret, thersh1 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
print(ret)
plt.smshow(thresh1, )