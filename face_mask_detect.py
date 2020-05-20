# importing all the necessary libraries
import os
import random
import shutil
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
from os import getcwd
from random import shuffle
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

# printing the total number of images in the dataset
print(len(os.listdir('/dataset/with_mask')))
print(len(os.listdir('/dataset/without_mask')))
# 1179
# 1055

# os.mkdir to create the directories
try:
  os.mkdir('/dataset/training')
  os.mkdir('/dataset/training/with-mask')
  os.mkdir('/dataset/training/without-mask')
  os.mkdir('/dataset/testing')
  os.mkdir('/dataset/testing/with-mask')
  os.mkdir('/dataset/testing/without-mask')
except OSError:
    print('error')

# os.listdir(DIRECTORY) gives you a listing of the contents of that directory
# os.path.getsize(PATH) gives you the size of the file
# copyfile(source, destination) copies a file from source to destination
def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
	all_images = os.listdir(SOURCE)
	shuffle(all_images)
	splitting_index = round(SPLIT_SIZE*len(all_images))
	train_images = all_images[:splitting_index]
	test_images = all_images[splitting_index:]

	# copy training images
	for img in train_images:
	src = os.path.join(SOURCE, img)
	dst = os.path.join(TRAINING, img)
	if os.path.getsize(src) <= 0:
		print(img+" is zero length, so ignoring!!")
	else:
	    shutil.copyfile(src, dst)

	# copy testing images
	for img in test_images:
	src = os.path.join(SOURCE, img)
	dst = os.path.join(TESTING, img)
	if os.path.getsize(src) <= 0:
	    print(img+" is zero length, so ignoring!!")
	else:
	    shutil.copyfile(src, dst)


WMASK_SOURCE_DIR = '/dataset/with_mask/'
WOMASK_SOURCE_DIR = '/dataset/without_mask/'
TRAINING_WMASK_DIR = '/dataset/training/with-mask/'
TRAINING_WOMASK_DIR = '/dataset/training/without-mask/'
TESTING_WMASK_DIR = '/dataset/testing/with-mask/'
TESTING_WOMASK_DIR = '/dataset/testing/without-mask/'

# 90% training data and 10% testing data
split_size = .9     
split_data(WMASK_SOURCE_DIR, TRAINING_WMASK_DIR, TESTING_WMASK_DIR, split_size)
split_data(WOMASK_SOURCE_DIR, TRAINING_WOMASK_DIR, TESTING_WOMASK_DIR, split_size)

# printing the len of the directories
print(len(os.listdir('/dataset/training/with-mask/')))
print(len(os.listdir('/dataset/training/without-mask/')))
print(len(os.listdir('/dataset/testing/with-mask/')))
print(len(os.listdir('/dataset/testing/without-mask/')))
# 1142
# 986
# 152
# 143

# model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])


TRAINING_DIR = '/dataset/training'
train_datagen = ImageDataGenerator(
    rescale=1 / 255,
    rotation_range=40,
    width_shift_range=.2,
    height_shift_range=.2,
    shear_range=.2,
    zoom_range=.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR,
    batch_size=15,
    class_mode='binary',
    target_size=(150, 150)
)

VALIDATION_DIR = '/dataset/testing'
validation_datagen = ImageDataGenerator(
    rescale=1 / 255,
    rotation_range=40,
    width_shift_range=.2,
    height_shift_range=.2,
    shear_range=.2,
    zoom_range=.2,
    horizontal_flip=True,
    fill_mode='nearest'

)
validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    batch_size=15,
    class_mode='binary',
    target_size=(150, 150)
)
# Found 2128 images belonging to 2 classes.
# Found 295 images belonging to 2 classes.

history = model.fit(train_generator,
                              epochs=100,
                              steps_per_epoch=17,
                              verbose=1,
                              validation_data=validation_generator,
                              validation_steps=17)


# PLOT LOSS AND ACCURACY

# Retrieve a list of list results on training and test data
# sets for each training epoch
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

# Plot training and validation accuracy per epoch
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")

plt.title('Training and validation accuracy')
plt.figure()

# Plot training and validation loss per epoch
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")

plt.title('Training and validation loss')
