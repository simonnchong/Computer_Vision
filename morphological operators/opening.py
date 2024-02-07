
#! useful for removing background noise, opening works as erosion -> dilation

import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_img():
    black_img = np.zeros((600,600))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(black_img, text="ABCDE", org=(50, 300), fontFace=font,
            fontScale=5, color=(255, 255, 255), thickness=30)
    return black_img

def display_img(img):  # this function just makes the image being displayed larger
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap="gray")
    
img = load_img()
# display_img(img)

kernel = np.ones((5,5), dtype=np.uint8)

#* add noises to the image
white_noise = np.random.randint(low=0,high=2,size=(600,600))
# display_img(white_noise)
#* convert the the white noise from 0-1 to 1-255, compatible with the image value 
white_noise = white_noise * 255

noise_img = img + white_noise
display_img(noise_img)

opening = cv2.morphologyEx(noise_img,cv2.MORPH_OPEN,kernel)
display_img(opening)


plt.show()
