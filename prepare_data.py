from pathlib import Path
import numpy
import pandas

# Cleaning divided_data directory
files = Path("divided_data").glob("**/**/*.jpg")
[[file.unlink() for file in files if file.is_file()]]

# Load training and test data into dataframes
data = pandas.read_csv("artist_artwork.csv", header=0)

# For training and validation set we take 80% of data
data_train = data.sample(frac=0.7, random_state=2)
# X forms the training images, and y forms the training labels
x_train = numpy.array(data_train.iloc[:, 1:])

# Here I split original training data to sub-training (80%) and validation data (20%)
# For testing set we take 20% of data
rest_data = data.drop(data_train.index)
data_val = rest_data.sample(frac=0.5, random_state=2)
# X_test forms the test images, and y_test forms the test labels
x_val = numpy.array(data_val.iloc[:, 1:])

# For testing set we take 20% of data
data_test = rest_data.drop(data_val.index)
# X_test forms the test images, and y_test forms the test labels
x_test = numpy.array(data_test.iloc[:, 1:])

# Moving proper data to proper directories
dirs = Path("divided_data").glob("**/**/")
# print([dir for dir in dirs])

for x in x_train:
    print(x[2])
