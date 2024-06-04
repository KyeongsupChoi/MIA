import os

def count_images(folder_path):
    # Define the image file extensions to look for
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')

    # Initialize a counter for the images
    image_count = 0

    # Walk through the folder and its subfolders
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_count += 1

    return image_count

# Specify the path to your folder
folder_path = '../../data/raw/img'

# Get the count of images
num_images = count_images(folder_path)

# Print the result
print(f'There are {num_images} images in the folder.')
