import shutil
from pathlib import Path

import numpy
import pandas
from tqdm import tqdm

# Prepare new CSV file with categories and artwork file name and write it in csv_data directory
print("Preparing new CSV")
csv_data = []
artists_counter = 0
artists = {}
for filename in tqdm(Path("resized").joinpath("resized").glob("*.jpg")):
    filename = filename.name
    if filename.count("_") == 3:
        class_name = "_".join(filename.split("_", 3)[:3])
    elif filename.count("_") == 2:
        class_name = "_".join(filename.split("_", 2)[:2])
    elif filename.count("_") == 1:
        class_name = "_".join(filename.split("_", 1)[:1])
    if class_name not in artists:
        artists[class_name] = artists_counter
        artists_counter += 1
    csv_data.append([artists[class_name], class_name, filename])
artist_artwork = pandas.DataFrame(csv_data, columns=["Category", "Name", "Artwork"])
artist_artwork.to_csv("csv_data/artist_artwork.csv")

# Copying divided_data_sceleton and renaming it to divided_data
shutil.copytree(
        Path("divided_data_sceleton"),
        Path("divided_data"),
    )

# Cleaning divided_data directory just in case
print("Cleaning directories so they are preapred for new datasets")
files = Path("divided_data").glob("**/**/*.jpg")
[[file.unlink() for file in tqdm(files) if file.is_file()]]

# Load training and test data into dataframes
data = pandas.read_csv("csv_data/artist_artwork.csv", header=0)
# For training set we take 70% of data
print(
    "Dividing data into three categories: training (70%), validation (15%) and test (15%)"
)
data_train = data.sample(frac=0.7, random_state=2)
# X forms the training images, and y forms the training labels
x_train = numpy.array(data_train.iloc[:, 1:])

# For validation set we take 15% of data
rest_data = data.drop(data_train.index)
data_val = rest_data.sample(frac=0.5, random_state=2)
# X_test forms the test images, and y_test forms the test labels
x_val = numpy.array(data_val.iloc[:, 1:])

# For testing set we take 15% of data
data_test = rest_data.drop(data_val.index)
# X_test forms the test images, and y_test forms the test labels
x_test = numpy.array(data_test.iloc[:, 1:])

# Moving proper data to proper directories
# Moving images chosen to training dataset
print("Preparing images for training dataset")
[
    shutil.copy(
        Path("images").joinpath("images").joinpath(x[1]).joinpath(x[2]),
        Path("divided_data").joinpath("x_train").joinpath(x[1]).joinpath(x[2]),
    )
    for x in tqdm(x_train)
]

# Moving images chosen to validation dataset
print("Preparing images for validation dataset")
[
    shutil.copy(
        Path("images").joinpath("images").joinpath(x[1]).joinpath(x[2]),
        Path("divided_data").joinpath("x_val").joinpath(x[1]).joinpath(x[2]),
    )
    for x in tqdm(x_val)
]

# Moving images chosen to test dataset
print("Preparing images for test dataset")
[
    shutil.copy(
        Path("images").joinpath("images").joinpath(x[1]).joinpath(x[2]),
        Path("divided_data").joinpath("x_test").joinpath(x[1]).joinpath(x[2]),
    )
    for x in tqdm(x_test)
]
print("All done and ready for preprocessing!")
