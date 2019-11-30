import csv
import os
from imageio import imread
import numpy as np



data_path = '../drivingData'
csv_fname = 'driving_log.csv'

# read annotation for data for images from csv file
lines = []
with open(os.path.join(data_path, csv_fname)) as csvf:
    reader = csv.reader(csvf)
    for line in reader:
        lines.append(line)

# print(len(lines))
# extract image path and read images
images = []
measurements = []
for line in lines:
    source_path = line[0]
    file_path = os.path.join('IMG', source_path.split('/')[-1])
    readable_path = os.path.join(data_path, file_path)
    image = imread(readable_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

# print(len(images))
# print(len(measurements))

X_train = np.array(images)
y_train = np.array(measurements)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

# model = Sequential()
# model.add(Flatten(input_shape=(160, 320, 3)))
# model.add(Dense(1)

model = Sequential()
input_shape = (160, 320, 3)
model.add(layers.Conv2D(3, (1, 1), 1,
                        padding="valid",
                        input_shape=input_shape))
# first convolution layer
model.add(layers.Conv2D(12, (5, 5), 1,
                        padding="valid",
                        input_shape=(32, 32, 3)))  # filter num, kernel size, stride, padding
model.add(layers.BatchNormalization(axis=-1))
model.add(layers.Activation("relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2), padding="valid"))
# second convolution layer, kernel_regularizer=regularizers.l2(0.1)
model.add(layers.Conv2D(32, (5, 5), 1,
                        padding="valid",
                        input_shape=(14, 14, 12)))
model.add(layers.BatchNormalization(axis=-1))
model.add(layers.Activation("relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2), padding="valid"))
model.add(layers.Flatten())
# first fully connected layer
model.add(layers.Dense(120))
model.add(layers.BatchNormalization())
model.add(layers.Activation("relu"))
model.add(layers.Dropout(0.5))
# second fully connected layer
model.add(layers.Dense(84))
model.add(layers.BatchNormalization())
# model.add(layers.Activation("relu"))
model.add(layers.Dropout(0.5))
# output layer
model.add(layers.Dense(1))
# model.add(layers.Activation("softmax"))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

model.save('model.h5')

