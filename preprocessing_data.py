# Imports
import os, pathlib
import pandas as pd
import cv2
import shutil

HEIGHT = 225
WIDTH = 225
IMAGE_FOLDER = "scaled_images"


# resizes images to Height and Width dimensions
def resize_sample(row: pd.Series) -> str:
    continent = row["continent"]
    image_path = row["path"]
    image_name = row["image_name"]

    image = cv2.imread(image_path)

    scaled_image = cv2.resize(image, (WIDTH, HEIGHT))

    path = f'{IMAGE_FOLDER}/{continent}'
    if not os.path.exists(path):
        os.mkdir(path)

    img_path = f'{path}/{image_name}'
    cv2.imwrite(img_path, scaled_image)
    return img_path


def main():

    # delete scaled_images folder
    if os.path.exists(IMAGE_FOLDER):
        shutil.rmtree(IMAGE_FOLDER)

    os.mkdir(IMAGE_FOLDER)

    df_metadata = pd.read_csv('country_data.csv')  # import metadata from append_metadata.py
    #df_metadata = df_metadata.iloc[:10, :] # for testing
    df_resized = df_metadata
    df_resized["path"] = df_metadata.apply(lambda sample: resize_sample(sample), axis=1)
    df_resized.to_csv("preprocessing_resized.csv", index=False)

if __name__ == "__main__":
    main()
