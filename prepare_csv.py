import os
import pandas

# Prepare new CSV file with categories and artwork file name and write it in csv_data directory
csv_data = []
artists_counter = 0
artists = {}
for dirname, _, filenames in os.walk("resized/resized"):
    for filename in filenames:
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
