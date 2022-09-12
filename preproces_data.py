# import keras, os
# from keras.models import Sequential
# from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.preprocessing.image import ImageDataGenerator

# import numpy

# Each image's dimension is 224 x 224
img_rows, img_cols = 224, 224

trdata = ImageDataGenerator()
train_data = trdata.flow_from_directory(
    directory="divided_data/x_train", target_size=(224, 224)
)
vdata = ImageDataGenerator()
val_data = vdata.flow_from_directory(
    directory="divided_data/x_val", target_size=(224, 224)
)
tsdata = ImageDataGenerator()
test_data = tsdata.flow_from_directory(
    directory="divided_data/x_test", target_size=(224, 224)
)
