from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


# Each image's dimension is 224 x 224
IMG_ROWS, IMG_COLS = 224, 224


def load_data():
    trdata = ImageDataGenerator(
        color_mode="rgb",
        shuffle=True,
        seed=42,
        vertical_flip=True,
        horizontal_flip=True,
        rotation_range=30,
        zoom_range=0.3,
        # brightness_range=[0.4,1.5]
        fill_mode="nearest",
        width_shift_range=0.2,
        height_shift_range=0.2,
    )

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15, 15))

    for i in range(4):
        # convert to unsigned integers for plotting
        image = next(train_generator)[0].astype("uint8")

        # changing size from (1, 200, 200, 3) to (200, 200, 3) for plotting the image
        image = np.squeeze(image)

        # plot raw pixel data
        ax[i].imshow(image)
        ax[i].axis("off")

    train_data = trdata.flow_from_directory(
        directory="divided_data/x_train",
        target_size=(IMG_ROWS, IMG_COLS),
        batch_size=32,
        class_mode="categorical",
    )
    vdata = ImageDataGenerator(
        vertical_flip=True,
    )
    val_data = vdata.flow_from_directory(
        directory="divided_data/x_val",
        target_size=(IMG_ROWS, IMG_COLS),
        batch_size=32,
        class_mode="categorical",
    )
    tsdata = ImageDataGenerator(
        vertical_flip=True,
    )
    test_data = tsdata.flow_from_directory(
        directory="divided_data/x_test",
        target_size=(IMG_ROWS, IMG_COLS),
        batch_size=32,
        class_mode="categorical",
    )
    return train_data, val_data, test_data
