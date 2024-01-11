# Imports
# Built-in
import os, shutil, pathlib

# External
import cv2
import pandas as pd

CROP_HEIGHT = 600
CROP_WIDTH = 600
CENTER_CROP = True
SCALE_HEIGHT = 224
SCALE_WIDTH = 224
IMAGE_FOLDER = "scaled_images"
CONTINENTS = ["Europe", "Asia"]
LIMIT_AMOUNT = True
IMAGE_COUNT = {"Europe":8000,"Asia":8000}
SEED = 42

# center crops images to maximum possible height and width within the specified values
def center_crop(img, dim):
	width, height = img.shape[1], img.shape[0]

	# process crop width and height for max available dimension
	crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
	crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
	mid_x, mid_y = int(width/2), int(height/2)
	cw2, ch2 = int(crop_width/2), int(crop_height/2) 
	crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
	return crop_img

# resizes images to Height and Width dimensions
def resize_sample(row: pd.Series) -> str:
    continent = row["continent"]
    image_path = row["path"]
    image_name = row["image_name"]

    image = cv2.imread(image_path)
    if CENTER_CROP:
        image = center_crop(image, (600,600))
    scaled_image = cv2.resize(image, (SCALE_WIDTH, SCALE_HEIGHT))

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
