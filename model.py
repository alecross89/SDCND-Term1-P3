#importing dependencies
import csv
import cv2
import os
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
import h5py
# print('Imports complete')

#importing the sample data from the driving log csv file
samples = []
with open('udacity_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# splitting data into validation and test sets
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# creating the generator 
def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			angles = []
			for batch_sample in batch_samples:
				for i in range(3):
					name = 'udacity_data/IMG/'+batch_sample[i].split('/')[-1]
					image = cv2.imread(name)
					angle = float(batch_sample[3])
					images.append(image)
					correction = 0.25
					#center image
					if i==0:
						angles.append(angle)
					#left image
					elif i==1:
						angles.append(angle + correction)
					#right image
					elif i==2:
						angles.append(angle - correction)
																		       
			#augmented data
			augmented_images = []
			augmented_angles = []
			for image, angle in zip(images, angles):
				augmented_images.append(image)
				augmented_angles.append(angle)
				flipped_image = cv2.flip(image, 1)
				flipped_angle = float(angle) * -1.0
				augmented_images.append(flipped_image)
				augmented_angles.append(flipped_angle)

			
			X_train = np.array(augmented_images)
			y_train = np.array(augmented_angles)
			yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# creating the model
model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))

model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Dropout(0.2))

model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# compiling the model
model.compile(loss='mse', optimizer='adam')

# fitting the model
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,
	nb_val_samples=len(validation_samples), nb_epoch=15)

# saving the model
model.save('model.h5')

print('model saved')