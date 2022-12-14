from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy
import calendar;
import time;

# Each image's dimension is 224 x 224
IMG_ROWS, IMG_COLS = 224, 224


def save_image_sample(train_data):
    # gmt stores current gmtime
    gmt = time.gmtime()
    
    # ts stores timestamp
    ts = calendar.timegm(gmt)
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15, 15))

    for i in range(4):
        # convert to unsigned integers for plotting
        image = next(train_data)[0].astype("uint8")
        image.shape

        # changing size from (1, 200, 200, 3) to (200, 200, 3) for plotting the image
        image = numpy.squeeze(image)

        # plot raw pixel data
        ax[i].imshow(image)
        ax[i].axis("off")
    fig.savefig('img.png')

def load_data(save_image=False):
    trdata = ImageDataGenerator(
        vertical_flip=True,
        horizontal_flip=True,
        rotation_range=0,
        zoom_range=0.5,
        brightness_range=None,
        fill_mode="nearest",
        width_shift_range=0.2,
        height_shift_range=0.2,
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-06,
        shear_range=0.0,
        channel_shift_range=0.0,
        cval=0.0,
        rescale=1./255.,
        preprocessing_function=None,
        data_format=None,
        validation_split=0.0,
        interpolation_order=1,
        dtype=None
    )
    train_data = trdata.flow_from_directory(
        directory="divided_data/x_train",
        target_size=(IMG_ROWS, IMG_COLS),
        batch_size=64,
        class_mode="categorical",
        color_mode="rgb",
        seed=2020,
        subset="training",
        shuffle=True,
    )


    if save_image:
        save_image_sample(train_data)
    
    vdata = ImageDataGenerator(
        vertical_flip=True,
        horizontal_flip=True,
        rotation_range=0,
        zoom_range=0.5,
        brightness_range=None,
        fill_mode="nearest",
        width_shift_range=0.2,
        height_shift_range=0.2,
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-06,
        shear_range=0.0,
        channel_shift_range=0.0,
        cval=0.0,
        rescale=1./255.,
        preprocessing_function=None,
        data_format=None,
        validation_split=0.0,
        interpolation_order=1,
        dtype=None
    )
    val_data = vdata.flow_from_directory(
        directory="divided_data/x_val",
        target_size=(IMG_ROWS, IMG_COLS),
        batch_size=64,
        class_mode="categorical",
        seed=2020,
        subset="validation",
        shuffle=True,
    )
    tsdata = ImageDataGenerator(
        vertical_flip=True,
        horizontal_flip=True,
        rotation_range=0,
        zoom_range=0.5,
        brightness_range=None,
        fill_mode="nearest",
        width_shift_range=0.2,
        height_shift_range=0.2,
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-06,
        shear_range=0.0,
        channel_shift_range=0.0,
        cval=0.0,
        rescale=1./255.,
        preprocessing_function=None,
        data_format=None,
        validation_split=0.0,
        interpolation_order=1,
        dtype=None
    )
    test_data = tsdata.flow_from_directory(
        directory="divided_data/x_test",
        target_size=(IMG_ROWS, IMG_COLS),
        batch_size=64,
        class_mode="categorical",
        seed=2020,
        subset="training",
        shuffle=True,
    )
    return train_data, val_data, test_data
