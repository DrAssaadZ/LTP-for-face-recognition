from Preprocessing import pre_processing
from LTP import ltp_indexation
from LTP import ltp_query
import os
import numpy as np
import cv2 as cv
import sys
import gc


# a function that creates the index base from an image directory
def create_index_base(path, gamma=0.2, alpha=0.10, tau=2.65, ltp_threshold=0.2):
    # listing all the images in the directory
    images_list = os.listdir(path)

    # initializing a list that contains the index base ([name of the image, image descriptor]
    index_base = []
    nbr_images = len(images_list)

    # for every images in the image base
    for i in range(nbr_images):

        # reading the image as a gray scale and normalizing it between 0 and 1
        img_full_path = path + '/' + images_list[i]
        img = cv.imread(img_full_path, cv.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

        # applying the pre-processing to the current image
        preprocessed_img = pre_processing(img, gamma, alpha, tau)

        # getting the image descriptor for the current image
        img_descriptor = ltp_indexation(preprocessed_img, ltp_threshold)

        # appending the image name and the descriptor to the index base list
        index_base.append([images_list[i], img_descriptor])

        # used for printing the progress on console
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%" % ('=' * (i // (nbr_images // 20) + 1), (i / nbr_images) * 100 + 1))
        sys.stdout.flush()

    # saving the index base as a numpy array
    index_base_np = np.array(index_base)
    # calling the garbage collector
    gc.collect()
    np.save('index_base.npy', index_base_np)


# a function that queries the index base and returns the n images with the shortest distance to the index base images
def interrogate_index_base(img, gamma=0.2, alpha=0.10, tau=2.65, ltp_threshold=0.2, ltp_tau=6, nbr_result=1):
    # loading the saved index base numpy array
    index_base = np.load('index_base.npy', allow_pickle=True)

    # list that contains the distances between the query image and the images in the index base
    image_distance = []

    # applying the pre-processing to the query image
    preprocessed_img = pre_processing(img, gamma, alpha, tau)

    # getting the upper and lower images for query image
    upper_descriptor, lower_descriptor = ltp_query(preprocessed_img, ltp_threshold)

    # uniform histogram possible values, taken from "https://en.wikipedia.org/wiki/Local_binary_patterns#Concept",
    uni_hist_keys = [0, 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48, 56, 60, 62, 63, 64, 96, 112, 120,
                     124, 126, 127, 128, 129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207, 223, 224, 225, 227, 231,
                     239, 240, 241, 243, 247, 248, 249, 251, 252, 253, 254, 255, 'others']

    # calculating the distance between the query image descriptor and the images descriptors in the index base
    for i in range(len(index_base)):
        # the sum of distances between the query image and index base images
        distance = 0

        # dividng the index base into upper base image descriptor and lower base image descriptor
        upper_base_img_descriptor = index_base[i, 1][0: 59]
        lower_base_img_descriptor = index_base[i, 1][59:]

        # initializing the upper and lower descriptors
        upper_uni_descriptor = dict(zip(uni_hist_keys, upper_base_img_descriptor))
        lower_uni_descriptor = dict(zip(uni_hist_keys, lower_base_img_descriptor))

        # looping through the size of the query image descriptor
        for w in range(1, (upper_descriptor.shape[0] - 1)):
            for h in range(1, (upper_descriptor.shape[1] - 1)):
                # getting the position of the current pixel in the upper and lower query images
                pixel_value_upper = upper_descriptor[w, h]
                pixel_value_lower = lower_descriptor[w, h]

                # checking if the upper or lower current pixel values are uniform
                # calculating the distance to the corresponding bin of the pixel value
                if pixel_value_upper in uni_hist_keys:
                    distance += min(upper_uni_descriptor[pixel_value_upper][w, h], ltp_tau)
                else:
                    distance += min(upper_uni_descriptor['others'][w, h], ltp_tau)
                # the same operations for the lower binary images
                if pixel_value_lower in uni_hist_keys:
                    distance += min(lower_uni_descriptor[pixel_value_lower][w, h], ltp_tau)
                else:
                    distance += min(lower_uni_descriptor['others'][w, h], ltp_tau)

        # appending the name of the image and the distance to the image distance list
        image_distance.append([index_base[i, 0], distance])

        # used for printing the progress on console
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%" % ('=' * (i // (len(index_base) // 20) + 1), (i / len(index_base)) * 100 + 1))
        sys.stdout.flush()

    # sorting the distances by the smallest distance
    ordered_img_distances = sorted(image_distance, key=lambda x: x[1])

    # returning the n images with the shortest distance to the index images
    return ordered_img_distances[0: nbr_result]









