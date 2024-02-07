import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_img():
    img = cv2.imread("../DATA/bricks.jpg").astype(np.float32) / 255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# print(load_img())


def display_img(img):  # this function just makes the image being displayed larger
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img)


i = load_img()

# * gamma is to adjust the brightness of the image
# * gamma = 4 # larger number makes the image darker
gamma = 1/4  # smaller number makes the image brighter
result = np.power(i, gamma)
# display_img(result)

img = load_img()
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img, text="bricks", org=(10, 600), fontFace=font,
            fontScale=10, color=(255, 0, 0), thickness=4)

# display_img(img)

# * this is a user defined kernel
kernel = np.ones(shape=(5, 5), dtype=np.float32) / 25
# print(kernel)

# * apply the kernel to the image
# * -1 is the depth that gonna apply to the image
blurred_img = cv2.filter2D(img, -1, kernel)
# display_img(blurred_img)

# * this is a built-in kernel
blurred_img2 = cv2.blur(img, ksize=(15, 15))
# display_img(blurred_img2)

# * this is a built-in kernel
blurred_img3 = cv2.GaussianBlur(img, (15, 15), 10)
# display_img(blurred_img3)

# * this is a built-in kernel
blurred_img4 = cv2.medianBlur(img, 5)
display_img(blurred_img4)

plt.show()
