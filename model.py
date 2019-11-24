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
from tensorflow.keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True)

model.save('model.h5')