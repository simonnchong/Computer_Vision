import cv2
import matplotlib.pyplot as plt

img = cv2.imread("../DATA/crossword.jpg", 0)
# plt.imshow(img, cmap="gray")

def show_pic(img):
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap="gray")

ret, output1 = cv2.threshold(img, 200,255,cv2.THRESH_BINARY)
show_pic(output)

output2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.thresh_bi)

plt.show()