from preproces_data import load_data
from tensorflow.keras.applications import inception_resnet_v2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import VGG19
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt

import os
import re
import glob
import hashlib
import argparse
import warnings

import six
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras import backend as K
from keras.layers import Dense, GlobalAveragePooling2D, Input, Flatten
from keras.applications import InceptionResNetV2
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping


IMAGE_SIZE = [224, 224]

train_data, val_data, test_data = load_data()
vggnet = InceptionResNetV2(
    input_shape=IMAGE_SIZE + [3], weights="imagenet", include_top=False
)
for layer in vggnet.layers:
    layer.trainable = False
x = Flatten()(vggnet.output)
prediction = Dense(train_data.num_classes, activation="softmax")(x)
model = Model(inputs=vggnet.input, outputs=prediction)
model.summary()


prediction = Dense(train_data.num_classes, activation="softmax")(x)
model.compile(
    loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
)

# fir model
r = model.fit_generator(
    train_data,
    validation_data=val_data,
    epochs=10,
    steps_per_epoch=len(train_data),
    validation_steps=len(val_data),
)


# plot the loss
plt.plot(r.history["loss"], label="train loss")
plt.plot(r.history["val_loss"], label="val loss")
plt.legend()
plt.show()
plt.savefig("LossVal_loss")

# plot the accuracy
plt.plot(r.history["accuracy"], label="train acc")
plt.plot(r.history["val_accuracy"], label="val acc")
plt.legend()
plt.show()
plt.savefig("AccVal_acc")
