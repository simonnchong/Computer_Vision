from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from numpy import genfromtxt

data = genfromtxt("../DATA/bank_note_data.txt", delimiter=",")

labels = data[:, 4]
features = data[:, 0:4]

X = features
y = labels

#! split the train and test dataset and labels
# * X = the dataset
# * y = the label

# * X_train = the 66% of training dataset
# * X_test = the 33% of the testing set
# * y_train = 66% of the label respective to the X_train
# * y_test = 33% of the label respective to the X_test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

#! to normalize the data into 0 - 1
scaler_object = MinMaxScaler()
scaler_object.fit(X_train)
scaled_X_train = scaler_object.transform(X_train)
scaled_X_test = scaler_object.transform(X_test)

#! configure the neural network layers
model = Sequential()

# * first hidden layer, 4 neurons, input dimension (bcs we have 4 columns in the dataset)
model.add(Dense(4, input_dim=4, activation="relu"))
model.add(Dense(8, activation="relu"))  # * second hidden layer
model.add(Dense(1, activation="sigmoid"))  # * 1 output bcs only either 0 or 1

model.compile(loss="binary_crossentropy", optimizer="adam", metrics="accuracy")
# * scaled_X_train = dataset, y_train = label for the dataset respectively
model.fit(scaled_X_train, y_train, epochs=10, verbose=2)

print(model.metrics_names)  # * to see the metrics being configured

#! evaluate the trained model
prediction = (model.predict(scaled_X_test) > 0.5).astype("int32")

print(confusion_matrix(y_test, prediction))  # to print the confusion matrix
print(classification_report(y_test, prediction))

#! save the trained model
model.save("my_model.h5")

#! load trained model
new_model = load_model("my_model.h5")

#! apply the trained model to new data
(new_model.predict(scaled_X_test) > 0.5).astype("int32")
