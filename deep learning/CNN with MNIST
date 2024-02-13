from keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from keras.models import Sequential
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.metrics import classification_report
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape)
single_image = x_train[0]  # * try to access the first image in dataset

# * show an example image from the MNIST dataset
# plt.imshow(single_image, cmap="gray")
# plt.imshow(single_image, cmap="gray_r") #* reverse color mapping

print(y_train)
# [5 0 4 ... 5 6 8]

# * normalize image value into 0 - 1, manually
x_train = x_train / 255
x_test = x_test / x_test.max()

# * convert the labels into one-hot encoding
# * e.g.: if the label for a particular image is "4", it will be [0,0,0,0,1,0,0,0,0,0,0], the index of "4" will be 1, which is index 5

y_cate_test = to_categorical(y_test, 10)  # * 10 number of classes
y_cate_train = to_categorical(y_train, 10)

# print(x_train[0])

# * this is just to define there is a channel in the training dataset
x_train = x_train.reshape(60000, 28, 28, 1)
# * this is just to define there is a channel in the testing dataset
x_test = x_test.reshape(10000, 28, 28, 1)


model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(4, 4),
          input_shape=(28, 28, 1), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())

# * hidden layer
model.add(Dense(128, activation="relu"))  # * neuron number

# * output layer
model.add(Dense(10, activation="softmax"))  # * 10 output classes

model.compile(loss="categorical_crossentropy",
              optimizer="rmsprop", metrics="accuracy")

model.summary()

model.fit(x_train, y_cate_train, epochs=2)

model.evaluate(x_test, y_cate_test)

# * let's the trained model predict the new test inpit
predictions = np.argmax(model.predict(x_test), axis=1)

print(y_test)  # * here using original the labels instead of one-hot encoding
print(predictions)  # * print the prediction label

print(classification_report(y_test, predictions))

plt.show()
