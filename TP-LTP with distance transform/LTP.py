import numpy as np
import cv2 as cv


# global variables
# vector used in converting from binary to decimal
conv_vector = [128, 64, 32, 1, 0, 16, 2, 4, 8]


# a method that calculates the distance transform of an image and returns a distance transform image
def distance_transform(img):

    return cv.distanceTransform(img, cv.DIST_L2, 0)


# LTP method used for the indexation phase, that takes an img and returns its descriptor
def ltp_indexation(img, threshold=0.2):

    # entered image dimensions
    img_height = img.shape[1]
    img_width = img.shape[0]

    # uniform histogram possible values, taken from "https://en.wikipedia.org/wiki/Local_binary_patterns#Concept",
    uni_hist_keys = [0, 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48, 56, 60, 62, 63, 64, 96, 112, 120,
                     124, 126, 127, 128, 129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207, 223, 224, 225, 227, 231,
                     239, 240, 241, 243, 247, 248, 249, 251, 252, 253, 254, 255, 'others']

    # variable used to create a dictionary which is of dimesion [possible uniform hist values, img_width, img_height]
    uni_hist_values = np.ones((59, img_width, img_height), dtype='uint8') * 255

    # initializing the upper and lower descriptors, a dictionary that contains uniform values as keys, and the sparse
    # binary images as values
    upper_descriptor = dict(zip(uni_hist_keys, uni_hist_values))
    lower_descriptor = dict(zip(uni_hist_keys, uni_hist_values))

    # looping through the pixels of the image
    for i in range(1, img_width - 1):
        for j in range(1, img_height - 1):
            # ltp_code is used to store the LTP ternary result code
            ltp_code = []
            # looping through the neighbours of the pixel
            for ww in range(i - 1, i + 2):
                for wh in range(j - 1, j + 2):
                    if img[ww, wh] > img[i, j] + threshold:
                        ltp_code.append(1)
                    elif img[ww, wh] < img[i, j] - threshold:
                        ltp_code.append(-1)
                    else:
                        ltp_code.append(0)

            # creating the LTP upper and lower codes
            ltp_upper_code = np.array(ltp_code)
            ltp_upper_code[ltp_upper_code == -1] = 0
            ltp_lower_code = np.array(ltp_code)
            ltp_lower_code[ltp_lower_code == 1] = 0

            # converting upper and lower pixel values from binary to decimal
            upper_pixel_value = np.sum(np.multiply(conv_vector, ltp_upper_code))
            lower_pixel_value = np.abs(np.sum(np.multiply(conv_vector, ltp_lower_code)))

            # creating the upper and lower binary images for upper and lower
            # if the value is uniform, set its location[i,j] in the upper binary image to 0 at that bin, otherwise
            # do the same for 'others' bin
            if upper_pixel_value in uni_hist_keys:
                upper_descriptor[upper_pixel_value][i, j] = 0
            else:
                upper_descriptor['others'][i, j] = 0

            # the same operations for the lower binary images
            if lower_pixel_value in uni_hist_keys:
                lower_descriptor[lower_pixel_value][i, j] = 0
            else:
                lower_descriptor['others'][i, j] = 0

    # creating the distance transform images for each possible value of the uniform histogram [59 distance transform
    # image for each histogram bin] for lower and upper
    for item in uni_hist_keys:
        upper_descriptor[item] = np.array(distance_transform(upper_descriptor[item]), dtype='uint8')
        lower_descriptor[item] = np.array(distance_transform(lower_descriptor[item]), dtype='uint8')

    # concatenating the upper and lower descriptors
    concatenated_descriptor = np.concatenate([list(upper_descriptor.values()), list(lower_descriptor.values())])

    return concatenated_descriptor


# LTP method used for the query phase, takes an image and returns 2 images, upper and lower pattern
def ltp_query(img, threshold=0.2):

    # entered image dimensions
    img_height = img.shape[1]
    img_width = img.shape[0]

    # initializing the upper and lower images
    upper_pattern = np.zeros((img_width, img_height))
    lower_pattern = np.zeros((img_width, img_height))

    # looping through the pixels of the image
    for i in range(1, img_width - 1):
        for j in range(1, img_height - 1):
            # ltp_code is used to store the LTP ternary result code
            ltp_code = []
            # looping through the neighbours of the pixel
            for ww in range(i - 1, i + 2):
                for wh in range(j - 1, j + 2):
                    if img[ww, wh] > img[i, j] + threshold:
                        ltp_code.append(1)
                    elif img[ww, wh] < img[i, j] - threshold:
                        ltp_code.append(-1)
                    else:
                        ltp_code.append(0)

            # creating the LTP upper and lower codes
            ltp_upper_code = np.array(ltp_code)
            ltp_upper_code[ltp_upper_code == -1] = 0
            ltp_lower_code = np.array(ltp_code)
            ltp_lower_code[ltp_lower_code == 1] = 0

            # converting upper and lower pixel values from binary to decimal
            upper_pixel_value = np.sum(np.multiply(conv_vector, ltp_upper_code))
            lower_pixel_value = np.abs(np.sum(np.multiply(conv_vector, ltp_lower_code)))

            # lower and upper pattern images
            upper_pattern[i, j] = upper_pixel_value
            lower_pattern[i, j] = lower_pixel_value

    return upper_pattern, lower_pattern


