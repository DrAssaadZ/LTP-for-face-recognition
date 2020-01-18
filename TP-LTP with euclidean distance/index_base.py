from Preprocessing import pre_processing
from LTP import ltp
import os
import numpy as np
import cv2 as cv
import sys
from scipy.spatial import distance


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
        img_descriptor = ltp(preprocessed_img, ltp_threshold)

        # appending the image name and the descriptor to the index base list
        index_base.append([images_list[i], img_descriptor])

        # used for printing the progress on console
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%" % ('=' * (i // (nbr_images // 20) + 1), (i / nbr_images) * 100 + 1))
        sys.stdout.flush()

    # saving the index base as a numpy array
    index_base_np = np.array(index_base)
    np.save('index_base.npy', index_base_np)


# a function that queries the index base and returns the n images with the shortest distance to the index base images
def interrogate_index_base(img, gamma=0.2, alpha=0.10, tau=2.65, ltp_threshold=0.2, nbr_result=1):
    # loading the saved index base numpy array
    index_base = np.load('index_base.npy', allow_pickle=True)

    # list that contains the distances between the query image and the images in the index base
    image_distance = []

    # applying the pre-processing to the query image
    preprocessed_img = pre_processing(img, gamma, alpha, tau)

    # getting the image descriptor of the query image
    img_descriptor = ltp(preprocessed_img, ltp_threshold)

    # calculating the distance between the query image descriptor and the images descriptors in the index base
    for i in range(len(index_base)):
        image_distance.append([index_base[i, 0], distance.euclidean(img_descriptor, index_base[i, 1])])

    # sorting the distances by the smallest distance
    ordered_img_distances = sorted(image_distance, key=lambda x: x[1])

    # returning the n images with the shortest distance to the index images
    return ordered_img_distances[0: nbr_result]









