# Import libraries
from logging.handlers import DatagramHandler
import numpy
import pandas

from keras.utils import to_categorical

# from sklearn.model_selection import train_test_split

# Load training and test data into dataframes
data = pandas.read_csv("artists.csv")

data_train = data.sample(frac=0.8, random_state=2)
data_test = data[~data.isin(data_train)]

# X forms the training images, and y forms the training labels
X = numpy.array(data_train.iloc[:, 1:])
y = to_categorical(numpy.array(data_train.iloc[:, 0]))

# # Here I split original training data to sub-training (80%) and validation data (20%)
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=13)

# # X_test forms the test images, and y_test forms the test labels
# X_test = numpy.array(data_test.iloc[:, 1:])
# y_test = to_categorical(numpy.array(data_test.iloc[:, 0]))
