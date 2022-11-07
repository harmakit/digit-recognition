import os
import cv2.cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt

from service.data_provider import DATADIR, DataProvider, DataProviderSource
from service.image_processor import ImageProcessor

dp = DataProvider(DataProviderSource.DATADIR)


def gif():
    image_processor = ImageProcessor()
    orig_path = DATADIR + "/orig"
    images = {}

    for i in range(10):
        digit_images_path = f'{orig_path}/../converted/{i}'
        digit_images_files = os.listdir(digit_images_path)
        # filter only .png files
        digit_images_files = list(filter(lambda x: x.endswith('.png'), digit_images_files))

        for digit_image_filename in digit_images_files:
            digit_image_path = f'{digit_images_path}/{digit_image_filename}'
            image = image_processor.get_image(digit_image_path)
            if i in images:
                images[i].append(image)
            else:
                images[i] = [image]

    avg_images = {
        0: [np.zeros((28, 28))],
        1: [np.zeros((28, 28))],
        2: [np.zeros((28, 28))],
        3: [np.zeros((28, 28))],
        4: [np.zeros((28, 28))],
        5: [np.zeros((28, 28))],
        6: [np.zeros((28, 28))],
        7: [np.zeros((28, 28))],
        8: [np.zeros((28, 28))],
        9: [np.zeros((28, 28))]
    }

    for j in range(10):
        for i in range(10):
            avg_images[i][0] = (avg_images[i][0] + images[i][j]) / 2

        show(avg_images, 'cum_' + str(j))


def show(images, title):
    fig, axs = plt.subplots(2, 5)
    for i in range(2):
        for j in range(5):
            digit = i * 5 + j
            axs[i, j].imshow(images[digit][0], cmap='gray')
    # plt.imsave('readme_images/avg_models/' + title + '.png', fig)
    plt.savefig('./readme/' + title + '.png')


gif()

# # average method
# def __init_avg_digits_data(digits_data) -> dict[int, np.ndarray]:
#     avg_digit_data = {}
#     lengths = [0] * 10
#     for i in range(len(digits_data)):
#         digit = digits_data[i][0]
#         current_avg_digit_data = avg_digit_data.get(digit, np.zeros((28, 28)))
#         avg_digit_data[digit] = current_avg_digit_data + digits_data[i][1]
#         lengths[digit] += 1
#
#     for i in range(10):
#         # calc average
#         avg_digit_data[i] = avg_digit_data[i] / lengths[i]
#         # convert to uint8
#         avg_digit_data[i] = avg_digit_data[i].astype(np.uint8)
#
#     return dict(sorted(avg_digit_data.items()))
