import cv2
import matplotlib.pyplot as plt

img = cv2.imread("DATA/rainbow.jpg", 0)

# plt.imshow(img, cmap="gray")

ret, thresh1 = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
# ret, thresh1 = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV) #invert the black and white color
# ret, thresh1 = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_TRUNC) #make the pixel over the threshold back to the threshold, otherwise remain the same
# ret, thresh1 = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_TOZERO) #make the pixel 
print(ret)

plt.imshow(thresh1, cmap="gray")
plt.show()