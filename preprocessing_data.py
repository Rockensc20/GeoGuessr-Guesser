# Imports
# Built-in
import os, shutil, pathlib

# External
import cv2
import pandas as pd

HEIGHT = 224
WIDTH = 224
IMAGE_FOLDER = "scaled_images"
CONTINENTS = ["Europe", "Asia"]
LIMIT_AMOUNT = True
IMAGE_COUNT = {"Europe":8000,"Asia":8000}
SEED = 42


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

def get_filtered_dataframe(df: pd.DataFrame):
    df_result = pd.DataFrame()
    if len(CONTINENTS) > 0: 
        df_filtered_continent = df[df['continent'].isin(CONTINENTS)]
    
        if LIMIT_AMOUNT: 
            for continent in CONTINENTS:
                df_sample = df_filtered_continent.groupby('continent').get_group(continent).sample(n=IMAGE_COUNT[continent], random_state=SEED)
                df_result= pd.concat([df_result,df_sample])
        else:
            df_result=df_filtered_continent
    else:
        df_result = df
    
    return df_result

def main():

    # delete scaled_images folder
    if os.path.exists(IMAGE_FOLDER):
        shutil.rmtree(IMAGE_FOLDER)
    os.mkdir(IMAGE_FOLDER)

    df_metadata = pd.read_csv('country_data.csv')  # import metadata from append_metadata.py, includes all countries and continents
    df_filtered_continent = get_filtered_dataframe(df_metadata)
    #df_metadata = df_metadata.iloc[:10, :] # for testing
    df_resized = df_filtered_continent
    df_resized["path"] = df_filtered_continent.apply(lambda sample: resize_sample(sample), axis=1)
    df_resized.to_csv("preprocessing_resized.csv", index=False)

if __name__ == "__main__":
    main()
