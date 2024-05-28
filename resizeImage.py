import os
from PIL import Image

def reduce_quality_image(imagePath, quality):

    image = Image.open(imagePath)
    image.save(image_path, quality=quality)

    return f"{imagePath} was reduced quality to {quality}"

def get_image_path_list_in_folder(folderPath):

    image_path_list = []
    for entry in os.listdir(folderPath):
        full_path = os.path.join(folderPath, entry)
        image_path_list.append(full_path)
    return image_path_list


if __name__ == '__main__':
    
    image_path_list = get_image_path_list_in_folder("FaceImages")

    for image_path in image_path_list :

        image_size_kb = os.path.getsize(image_path)/1000

        if image_size_kb > 300 :

            print(image_path)
            print(image_size_kb)

            reduce_quality_image(image_path, quality=80)



