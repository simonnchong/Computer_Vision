from keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from keras.models import Sequential
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.datasets import cifar10
import numpy as np
from keras.models import load_model


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape)  # * (50000, 32, 32, 3)
print(x_train[0].shape)  # * (32, 32, 3)


print(y_train.max())  # * there are total 10 classes
print(y_train.min())
# plt.imshow(x_train[2])

x_train = x_train / 255
x_test = x_test / 255

print(x_test.shape)  # * (10000, 32, 32, 3)

# * one-hot encoding
y_cate_train = to_categorical(y_train, 10)
y_cate_test = to_categorical(y_test, 10)


model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(32, 32, 3), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(32, 32, 3), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(32, 32, 3), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())

# * usually the neuron number would be 2^x
model.add(Dense(256, activation="relu"))  # * neuron number

# * output layer
model.add(Dense(10, activation="softmax"))  # * 10 output classes


model.compile(loss="categorical_crossentropy",
              optimizer="rmsprop", metrics="accuracy")

model.summary()

model.fit(x_train, y_cate_train, epochs=1)
model.evaluate(x_test, y_cate_test)

from sklearn.metrics import classification_report

predictions = np.argmax(model.predict(x_test), axis=1)
print(classification_report(y_test,predictions))

#! save the trained model
model.save("cifar10.h5")

#! load trained model
new_model = load_model("cifar10.h5")

#! apply the trained model to new data
np.argmax(model.predict(x_test), axis=1)

plt.show()
