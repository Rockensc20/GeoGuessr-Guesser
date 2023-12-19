# Imports
# Built-in
import os, shutil, pathlib

# External
import pandas as pd
from PIL import Image
import pycountry_convert as pc


# Helper function for converting the folder name of each country to the respective continent it is in
def country_to_continent(country_name):
    
    country_alpha2 = pc.country_name_to_country_alpha2(country_name)
    
    # Handle exceptions that are not part of the package
    if country_alpha2 == 'AQ':
        return 'Antarctica'
    
    country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
    country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
    return country_continent_name

def get_metadata_per_country(country_path):
    metadata_country = []
    for file_path in country_path.glob('*'):
        # Get the width and height of the image
        width, height = Image.open(file_path).size
        
        metadata_country.append({
            'country': country_path.name,
            'continent': country_to_continent(country_path.name),
            'image_name': file_path.name,
            'width': width,
            'height': height,
            'size': file_path.stat().st_size,
            'path': file_path
        })
    
    return metadata_country

def get_metadata_for_folder(dir_path):
    working_dir = pathlib.Path().absolute()
    dataset_dir=pathlib.Path(os.path.join(working_dir, 'compressed_dataset'))
   # Get all country folders
    country_folders = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]

    # Initialize an empty list to store metadata
    all_metadata = []

    # Iterate through country folders and extract metadata
    for country_name in country_folders:
        metadata_country = get_metadata_per_country(pathlib.Path(os.path.join(dataset_dir, country_name)))
        all_metadata.extend(metadata_country)

    # Create a Pandas DataFrame from the metadata
    df_geo_data = pd.DataFrame(all_metadata)
    #df_data_distribution = df_geo_data.groupby('country')['image_name'].count().reset_index().rename(columns={'image_name': 'frequency'})

    df_geo_data.to_csv('country_data.csv', index=False)
    return print(df_geo_data)

def main():
    working_dir = pathlib.Path().absolute()
    dataset_dir=pathlib.Path(os.path.join(working_dir, 'compressed_dataset'))
    get_metadata_for_folder(dataset_dir)

if __name__ == "__main__":
    main()
