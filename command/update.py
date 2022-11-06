import os
import cv2.cv2 as cv2
import numpy as np
from service.data_provider import DATADIR
from service.image_processor import ImageProcessor


def update():
    image_processor = ImageProcessor()
    orig_path = DATADIR + "/orig"
    for i in range(10):
        digit_images_path = f'{orig_path}/{i}'
        digit_images_files = os.listdir(digit_images_path)
        digit_images = []
        for digit_image_filename in digit_images_files:
            digit_image_path = f'{digit_images_path}/{digit_image_filename}'
            converted_image_path = f'{DATADIR}/converted/{i}/{digit_image_filename}'

            image = image_processor.get_image(digit_image_path)
            cv2.imwrite(converted_image_path, image)

            digit_images.append(image)

        digit_images = np.array(digit_images)
        np.save(f'{DATADIR}/npy/digit_{i}.npy', digit_images)
        print(f'Updated digit {i} with {len(digit_images)} images')

    print('Updated data')
