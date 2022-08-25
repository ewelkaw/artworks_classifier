import os
from logging.handlers import DatagramHandler
import numpy
import pandas
from tensorflow.keras.utils import to_categorical
import keras
from sklearn.model_selection import train_test_split

# Load training and test data into dataframes
data = pandas.read_csv("artist_artwork.csv")

# For training and validation set we take 80% of data
data_train = data.sample(frac=0.8, random_state=2)
# X forms the training images, and y forms the training labels
data_train_values = numpy.array(data_train.iloc[:, 1:])
data_train_labels = to_categorical(numpy.array(data_train.iloc[:, 0], dtype=int))

# Here I split original training data to sub-training (80%) and validation data (20%)
x_train, x_val, y_train, y_val = train_test_split(
    data_train_values, data_train_labels, test_size=0.2, random_state=13
)

# For testing set we take 20% of data
data_test = data[~data.isin(data_train)]
# X_test forms the test images, and y_test forms the test labels
x_test = numpy.array(data_test.iloc[:, 1:])
y_test = to_categorical(numpy.array(data_test.iloc[:, 0], dtype=int))


# Each image's dimension is 224 x 224
img_rows, img_cols = 224, 224
