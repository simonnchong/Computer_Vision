import cv2
import matplotlib.pyplot as plt

img = cv2.imread("../DATA/crossword.jpg", 0)
# plt.imshow(img, cmap="gray")

def show_pic(img):
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap="gray")

ret, output1 = cv2.threshold(img, 200,255,cv2.THRESH_BINARY)
show_pic(output1)

output2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,8)
show_pic(output2)

blended = cv2.addWeighted(src1=output1, alpha=0.6, src2=output2, )

plt.show()