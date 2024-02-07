
import cv2
import numpy as np
import matplotlib.pyplot as plt


def display_img(img):  # this function just makes the image being displayed larger
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap="gray")


gorilla = cv2.imread("../DATA/gorilla.jpg", 0)
# display_img(gorilla)

hist_value = cv2.calcHist([gorilla], channels=[0],
                          mask=None, histSize=[256], ranges=[0, 256])
# plt.plot(hist_value)

equalized_gorilla = cv2.equalizeHist(gorilla)

# display_img(equalized_gorilla)
hist_value = cv2.calcHist([equalized_gorilla], channels=[
                          0], mask=None, histSize=[256], ranges=[0, 256])
# plt.plot(hist_value)

color_gorilla = cv2.imread("../DATA/gorilla.jpg")
show_gorilla = cv2.cvtColor(color_gorilla, cv2.COLOR_BGR2RGB)

#! equalize color image
display_img(show_gorilla)

# * 1. convert the RGB image into HSV format
hsv_gorilla = cv2.cvtColor(color_gorilla, cv2.COLOR_BGR2HSV)

# * 2. hue:saturation:value. we only equalize the value channel then replace the original value in the image
hsv_gorilla[:, :, 2] = cv2.equalizeHist(hsv_gorilla[:, :, 2])

# * 3. convert back to RGB format
equalized_color_gorilla = cv2.cvtColor(hsv_gorilla, cv2.COLOR_HSV2RGB)
display_img(equalized_color_gorilla)

plt.show()
