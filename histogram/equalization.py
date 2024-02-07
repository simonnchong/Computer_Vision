
import cv2
import numpy as np
import matplotlib.pyplot as plt

rainbow = cv2.imread("../DATA/rainbow.jpg")
show_rainbow = cv2.cvtColor(rainbow, cv2.COLOR_BGR2RGB)

mask = np.zeros(rainbow.shape[:2],np.uint8)
# plt.imshow(mask,cmap="gray")

mask[300:400,100:400] = 255
# plt.imshow(mask,cmap="gray")

masked_rainbow = cv2.bitwise_and(rainbow,rainbow,mask=mask)
show_masked_rainbow = cv2.bitwise_and(show_rainbow,show_rainbow,mask=mask)
# plt.imshow(show_masked_rainbow)

hist_masked_values_red = cv2.calcHist([rainbow], channels=[2], mask=mask,histSize=[256],ranges=[0,256])
hist_values_red = cv2.calcHist([rainbow], channels=[2], mask=None,histSize=[256],ranges=[0,256])

plt.plot(hist_masked_values_red)
plt.title("Red histogram for masked rainbow")
# plt.plot(hist_values_red)

plt.show()