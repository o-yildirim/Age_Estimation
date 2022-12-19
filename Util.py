import cv2
import os
import numpy as np

def get_data(path, number_of_images_to_retrieve):  # Returns images and labels as two seperate numpy arrays.
    file_names = sorted(os.listdir(path))
    if number_of_images_to_retrieve <= 0:  # Gets the first "number_of_images_to_retrieve" images from the specified path. If <= 0, returns the entire data in that path.
        number_of_images_to_retrieve = len(file_names)

    images = []
    labels = np.empty(0)
    file_names_ret = []
    number_count = 0
    for file_name in file_names:
        img = cv2.imread(os.path.join(path, file_name))
        if img is not None:
            images.append(img / 255)
            label = int(file_name.split('_')[0].lstrip('0'))  # Label
            current_file_name = file_name.split('_')[1]
            labels = np.append(labels, label)
            file_names_ret = np.append(file_names_ret, current_file_name)
        number_count += 1
        if number_count >= number_of_images_to_retrieve:
            break
    images = np.array(images)
    images = preprocess_data(images)
    return images, labels, file_names_ret


def preprocess_data(images):
    mean_image = np.mean(images, axis=0)
    preprocessed_images = []
    for image in images:
        preprocessed_images.append(np.subtract(image, mean_image))
    return np.array(preprocessed_images)