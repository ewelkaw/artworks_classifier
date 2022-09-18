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
from keras.layers import Dense, GlobalAveragePooling2D, Input, Flatten, BatchNormalization, Activation
from keras.applications import InceptionResNetV2, ResNet50, ResNet152
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
import calendar;
import time;
from plot_results import plot_results
from prepare_class_weights import prepare_class_weights


# Each image's dimension is 224 x 224
IMG_ROWS, IMG_COLS = 224, 224

train_data, val_data, test_data = load_data()

STEP_SIZE_TRAIN = train_data.n//train_data.batch_size
STEP_SIZE_VALID = val_data.n//val_data.batch_size

class_weights = prepare_class_weights()

base_model = ResNet152(weights='imagenet', include_top=False, pooling ="max",input_shape=(IMG_ROWS, IMG_COLS,3))

for layer in base_model.layers:
    layer.trainable = False

# Adding some more layers
x = Flatten()(base_model.output)
prediction = Dense(train_data.num_classes, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=prediction)
model.summary()

optimizer = Adam(learning_rate=0.0001)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer, 
              metrics=['accuracy'])

n_epoch = 20

early_stop = EarlyStopping(monitor='loss', patience=20, verbose=1, 
                           mode='auto', restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, 
                              verbose=1, mode='auto')

model.compile(
    loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
)

# fit model
r = model.fit(
    train_data,
    epochs=n_epoch,
    validation_data=val_data,
    callbacks=[reduce_lr, early_stop],
    verbose=1,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_steps=STEP_SIZE_VALID,
)

# gmt stores current gmtime
gmt = time.gmtime()
  
# ts stores timestamp
ts = calendar.timegm(gmt)

# # plot the loss
# plt.plot(r.history["loss"], label="train loss")
# plt.plot(r.history["val_loss"], label="val loss")
# plt.legend()
# plt.savefig(f"loss_{ts}")

# # plot the accuracy
# plt.plot(r.history["accuracy"], label="train acc")
# plt.plot(r.history["val_accuracy"], label="val acc")
# plt.legend()
# plt.savefig(f"accuracy_{ts}")


# # Freeze core ResNet layers and train again 
# for layer in model.layers[-6:]:
#    layer.trainable = False

# # for layer in model.layers:
# #     layer.trainable = True

# optimizer = Adam(learning_rate=0.0001)
# model.compile(loss='categorical_crossentropy',
#               optimizer=optimizer, 
#               metrics=['accuracy'])

# n_epoch = 25
# history2 = model.fit_generator(generator=train_data, steps_per_epoch=STEP_SIZE_TRAIN,
#                               validation_data=val_data, validation_steps=STEP_SIZE_VALID,
#                               epochs=n_epoch,
#                               shuffle=True,
#                               verbose=1,
#                               callbacks=[reduce_lr, early_stop],
#                               use_multiprocessing=True,
#                               workers=16,
#                               class_weight=class_weights
#                              )

history = {}
history['loss'] = r.history['loss']
history['acc'] = range.history['accuracy']
history['val_loss'] = r.history['val_loss']
history['val_acc'] = r.history['val_accuracy']
history['learning_rate'] = r.history['learning_rate']


plot_results(history)

# Prediction accuracy on training data
score = model.evaluate_generator(train_data)
print("Prediction accuracy on train data =", score[1])

# Prediction accuracy on validation data
score = model.evaluate_generator(val_data)
print("Prediction accuracy on validation data =", score[1])

# Prediction accuracy on test data
score = model.evaluate_generator(test_data)
print("Prediction accuracy on test data =", score[1])
# vggnet = InceptionResNetV2(
#     input_shape=IMAGE_SIZE + [3], weights="imagenet", include_top=False
# )
# for layer in vggnet.layers:
#     layer.trainable = False
# x = Flatten()(vggnet.output)
# prediction = Dense(train_data.num_classes, activation="softmax")(x)
# model = Model(inputs=vggnet.input, outputs=prediction)
# model.summary()


# prediction = Dense(train_data.num_classes, activation="softmax")(x)
# model.compile(
#     loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
# )

# # fit model
# r = model.fit(
#     train_data,
#     validation_data=val_data,
#     epochs=10,
#     steps_per_epoch=len(train_data),
#     validation_steps=len(val_data),
# )

# # gmt stores current gmtime
# gmt = time.gmtime()
  
# # ts stores timestamp
# ts = calendar.timegm(gmt)

# # plot the loss
# plt.plot(r.history["loss"], label="train loss")
# plt.plot(r.history["val_loss"], label="val loss")
# plt.legend()
# plt.savefig(f"loss_{ts}")

# # plot the accuracy
# plt.plot(r.history["accuracy"], label="train acc")
# plt.plot(r.history["val_accuracy"], label="val acc")
# plt.legend()
# plt.savefig(f"accuracy_{ts}")
