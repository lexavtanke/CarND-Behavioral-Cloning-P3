import csv
import os
from imageio import imread
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Flatten, Dense, Lambda, Cropping2D

# create dataframe for images and steering angles  
df = pd.DataFrame(columns=['image', 'measure'])
# training data collected during several session
# so we need to read all data csv
data_path = '../drivingData'
csv_fname = 'driving_log.csv'
data_files = [] 

for file in os.listdir(data_path):
    if file.endswith(".csv"):
        data_files.append(file)
print(f'There are files {data_files} in data_path')

# use all three cameras      
for file in data_files:
    with open(os.path.join(data_path, file)) as csvf:
        next(csvf)
        reader = csv.reader(csvf)
        for line in reader:
            for i in range(3):
                source_path = line[i]
#                     source_path = line[0]
                file_path = source_path.split('/')[-1]
        #         print(file_path)

        #         print(measurement)
        #         measurements.append(measurement)
                if i == 0:
                    measurement = float(line[3])
                elif i == 1:
                    measurement = float(line[3]) + 0.1
                elif i == 2:
                    measurement = float(line[3]) - 0.1
                nano_df = pd.DataFrame([[file_path, measurement]], columns=['image', 'measure'])
                df = df.append(nano_df, ignore_index=True)
                
# show dataframe info if needed
# df.info()

# add some normalization to the data and cut some really strait steering angles examples
# to make data normaly distributed
norm_df= []
norm_df = df.drop(df[(df['measure'] < 0.02) & (df['measure'] > -0.01)].index)
norm_df = norm_df.drop(norm_df[(norm_df['measure'] < -0.099) & (norm_df['measure'] > -0.12)].index)
norm_df = norm_df.drop(norm_df[(norm_df['measure'] < 0.12) & (norm_df['measure'] > 0.095)].index)

# define net
model_name = "model" 
# Nvidia net without batch normalization
model = Sequential()
input_shape = (80, 160, 3)
model.add(Cropping2D(cropping=((25,10), (0,0)), input_shape=input_shape))
model.add(Lambda(lambda x: tf.cast(x, tf.float32) / 255.0 - 0.5))
# first convolution layer
model.add(layers.Conv2D(24, (5, 5), 1,
                        padding="valid"))  # filter num, kernel size, stride, padding
model.add(layers.BatchNormalization())
model.add(layers.Activation("relu"))
# second convolution layer, kernel_regularizer=regularizers.l2(0.1)
model.add(layers.Conv2D(36, (5, 5), 2,
                        padding="valid"))
model.add(layers.BatchNormalization())
model.add(layers.Activation("relu"))
# third conv layer
model.add(layers.Conv2D(48, (3, 3), 2,
                        padding="valid"))
model.add(layers.BatchNormalization())
model.add(layers.Activation("relu"))
# forth conv layer
model.add(layers.Conv2D(64, (3, 3), 1,
                        padding="valid"))
model.add(layers.BatchNormalization())
model.add(layers.Activation("relu"))
model.add(layers.Flatten())
# first fully connected layer
model.add(layers.Dense(120))
model.add(layers.BatchNormalization())
model.add(layers.Activation("relu"))
model.add(layers.Dropout(0.5))
# second fully connected layer
model.add(layers.Dense(84))
model.add(layers.BatchNormalization())
model.add(layers.Activation("relu"))
model.add(layers.Dropout(0.5))
# third fully connected layer
model.add(layers.Dense(10))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
# output layer
model.add(layers.Dense(1))

# define generator for learing
BS = 100
target_size = (80, 160)
dataframe = norm_df
directory = data_path
# define data to train and validation
datagen=ImageDataGenerator(validation_split=0.2)
train_generator=datagen.flow_from_dataframe(dataframe=dataframe,
                                            directory=directory,
                                            x_col="image",
                                            y_col="measure",
                                            subset="training",
                                            batch_size=BS,
                                            seed=42,
                                            shuffle=True,
                                            class_mode="raw",
                                            target_size=target_size)

valid_generator=datagen.flow_from_dataframe(dataframe=dataframe,
                                            directory=directory,
                                            x_col="image",
                                            y_col="measure",
                                            subset="validation",
                                            batch_size=BS,
                                            seed=42,
                                            shuffle=True,
                                            class_mode="raw",
                                            target_size=target_size)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

NUM_EPOCHS = 20
INIT_LR = 2*1e-3
        
opt = Adam(lr=INIT_LR, decay=INIT_LR / (NUM_EPOCHS * 0.5))
model.compile(loss="mse", optimizer=opt, metrics=["mse"])

# use checkpointer to save best model by validation mean squared error
checkpointer = ModelCheckpoint(filepath=f"{model_name}.h5", 
                               monitor = 'val_mse',
                               verbose=1, 
                               save_best_only=True)

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

# train model
H = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=NUM_EPOCHS,
                    callbacks=[checkpointer]
)
