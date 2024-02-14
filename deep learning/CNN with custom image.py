from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, Activation, Dropout, MaxPool2D, Dense, Flatten
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.metrics import classification_report
import numpy as np
from keras.utils import to_categorical
from keras.models import load_model
import cv2

# * show example from the dataset
cat4 = cv2.imread("../CATS_DOGS/train/CAT/4.jpg")
cat4 = cv2.cvtColor(cat4, cv2.COLOR_BGR2RGB)
dog2 = cv2.imread("../CATS_DOGS/train/DOG/2.jpg")
dog2 = cv2.cvtColor(dog2, cv2.COLOR_BGR2RGB)

# plt.imshow(dog2)0


image_gen = ImageDataGenerator(rotation_range=30,  # * degree range for random rotations.
                               width_shift_range=0.1,  # * decrease height up to 10%
                               height_shift_range=0.1,  # * increase height up to 10%
                               rescale=1/255,  # * here means to multiply 255
                               # * shear Intensity (Shear angle in counter-clockwise direction in degrees)
                               shear_range=0.1,
                               zoom_range=0.2,
                               horizontal_flip=True,
                               fill_mode="nearest")  # * fill in the space when enlarge/rescale the image

# * check the output of augmented dataset
# plt.imshow(image_gen.random_transform(dog2))


model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(150, 150, 3), activation="relu"))  # * since the image is now an object of ImageDataGenerator, so they will be adjusted automatically when setting the input_shape here
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(150, 150, 3), activation="relu")) 
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(150, 150, 3), activation="relu")) 
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())

# * hidden layer
model.add(Dense(128, activation="relu"))  # * 128 is the neuron number
# model.add(Dense(128))  # * same meaning as previous line
# model.add(Activation("relu")) # * same meaning as previous line

model.add(Dropout(0.5)) #* turn off 50% of the neuron randomly during training

# * output layer
model.add(Dense(1, activation="sigmoid"))  # * 10 output classes
model.add(Dense(1)) #* only 1 output, car:0 or dog:1
model.add(Activation("sigmoid")) #* same meaning as previous line

model.summary()
model.compile(loss="binary_crossentropy",
              optimizer="adam", metrics=["accuracy"])

batch_size = 16 #* number of batch of image into a single step in epoch, no right or wrong number

train_img_gen = image_gen.flow_from_directory("../CATS_DOGS/train", target_size=(150,150), batch_size=batch_size,class_mode="binary")
test_img_gen = image_gen.flow_from_directory("../CATS_DOGS/test", target_size=(150,150), batch_size=batch_size,class_mode="binary")

#* can check the label like this
print(train_img_gen.class_indices)
#* {'CAT': 0, 'DOG': 1}

#* steps_per_epoch is to reduce the processing all images, here 150 x 16 (batch size), so training data will be 2400 images
results = model.fit_generator(train_img_gen,epochs=1,steps_per_epoch=150,validation_data=test_img_gen,validation_steps=12)


plt.show()
