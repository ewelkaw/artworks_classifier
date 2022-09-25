from preproces_data import load_data
from keras.models import Model
from keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    Input,
    Flatten,
    BatchNormalization,
    Activation,
)
from keras.applications import ResNet152
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import calendar
import time
from plot_results import plot_results
from prepare_class_weights import prepare_class_weights


# Each image's dimension is 224 x 224
IMG_ROWS, IMG_COLS = 224, 224

train_data, val_data, test_data = load_data()

STEP_SIZE_TRAIN = train_data.n // train_data.batch_size
STEP_SIZE_VALID = val_data.n // val_data.batch_size

class_weights = prepare_class_weights()

base_model = ResNet152(
    weights="imagenet",
    include_top=False,
    pooling="max",
    input_shape=(IMG_ROWS, IMG_COLS, 3),
)
base_model.trainable = False

# create new model at the top
# inputs = Input(shape=(IMG_ROWS, IMG_COLS, 3))
# We make sure that the base_model is running in inference mode here,
# by passing `training=False`. This is important for fine-tuning, as you will
# learn in a few paragraphs.
# x = base_model(inputs, training=False)
x = Flatten()(base_model.output)
# Convert features of shape `base_model.output_shape[1:]` to vectors
x = GlobalAveragePooling2D()(x)
# A Dense classifier with a single unit (binary classification)
outputs = Dense(train_data.num_classes, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=outputs)
model.summary()


# Adding some more layers
# x = Flatten()(base_model.output)
# prediction = Dense(train_data.num_classes, activation="softmax")(x)
# model = Model(inputs=base_model.input, outputs=prediction)
# model.summary()

optimizer = Adam(learning_rate=0.0001)

n_epoch = 20

early_stop = EarlyStopping(
    monitor="loss", patience=20, verbose=1, mode="auto", restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="loss", factor=0.1, patience=5, verbose=1, mode="auto"
)

model.compile(
    loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
)

# fit model
result = model.fit(
    train_data,
    epochs=n_epoch,
    validation_data=val_data,
    callbacks=[reduce_lr, early_stop],
    verbose=1,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_steps=STEP_SIZE_VALID,
    class_weights=class_weights,
)

# gmt stores current gmtime
gmt = time.gmtime()

# ts stores timestamp
ts = calendar.timegm(gmt)

history = {
    "loss": result.history["loss"],
    "acc": result.history["accuracy"],
    "val_loss": result.history["val_loss"],
    "val_acc": result.history["val_accuracy"],
    "learning_rate": result.history["learning_rate"],
}

plot_results(history, ts)

# Prediction accuracy on training data
train_data_prediction = model.evaluate(train_data)
print("Prediction accuracy on train data =", train_data_prediction[1])

# Prediction accuracy on validation data
val_data_prediction = model.evaluate(val_data)
print("Prediction accuracy on validation data =", val_data_prediction[1])

# Prediction accuracy on test data
test_data_prediction = model.evaluate(test_data)
print("Prediction accuracy on test data =", test_data_prediction[1])
