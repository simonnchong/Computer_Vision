
import cv2
import numpy as np
import matplotlib.pyplot as plt

horse = cv2.imread("../DATA/horse.jpg") #* original BGR OpenCV for operation
show_horse = cv2.cvtColor(horse, cv2.COLOR_BGR2RGB) #* converted into RGB to display

rainbow = cv2.imread("../DATA/rainbow.jpg")
show_rainbow = cv2.cvtColor(rainbow, cv2.COLOR_BGR2RGB)

bricks = cv2.imread("../DATA/bricks.jpg")
show_bricks = cv2.cvtColor(bricks, cv2.COLOR_BGR2RGB)

# plt.imshow(show_horse)

#! this is to display only 1 color channel
#* BGR order in openCV
#* channels=[0] means Blue, [1] means Green, [2] means Red 
#* histSize=[256] bcs we have 0-255 (total 256 pixel value)
#* ranges=[0,256] bcs we have 0-255 (256 is exclusive)
hists_value = cv2.calcHist([horse],channels=[0], mask=None,histSize=[256], ranges=[0,256]) 

# plt.plot(hists_value)

img = bricks
color = ("b", "g", "r")

for index, colors in enumerate(color):
    histr = cv2.calcHist([img],[index],None,[256],[0,256])
    plt.plot(histr,color=colors)
    plt.xlim([0,256]) #*xlim means x limit for the x-axis in the histogram, smaller number means zoom the x-axis, same goes to y-axis 
    
plt.title("Histogram of the image")

plt.show()