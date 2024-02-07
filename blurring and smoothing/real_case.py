
#! reduce the noise from the image

import cv2
import numpy as np
import matplotlib.pyplot as plt

noised_img = cv2.imread("../DATA/sammy_noise.jpg")
noised_img = cv2.cvtColor(noised_img, cv2.COLOR_BGR2RGB)


def display_img(img):  # this function just makes the image being displayed larger
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img)


display_img(noised_img)

# apply median blur to noised image
median = cv2.medianBlur(noised_img, 5)
display_img(median)

plt.show()
