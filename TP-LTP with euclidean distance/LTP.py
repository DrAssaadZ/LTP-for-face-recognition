import numpy as np


# a method that applies the LTP method on a given image, and returns its descriptor (uniform histogram) of size 118
def ltp(img, threshold=0.2):

    # entered image dimensions
    img_height = img.shape[1]
    img_width = img.shape[0]

    # vector used in converting from binary to decimal
    conv_vector = [128, 64, 32, 1, 0, 16, 2, 4, 8]

    '''
    initializing upper and lower patterns, this block of code is commented because it is not needed in our code
    its main purpose in only to display the upper and lower image results
    '''
    # upper_pattern = np.zeros((img_width, img_height))
    # lower_pattern = np.zeros((img_width, img_height))

    # uniform histogram possible values, taken from "https://en.wikipedia.org/wiki/Local_binary_patterns#Concept",
    uni_hist_keys = [0, 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48, 56, 60, 62, 63, 64, 96, 112, 120,
                     124, 126, 127, 128, 129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207, 223, 224, 225, 227, 231,
                     239, 240, 241, 243, 247, 248, 249, 251, 252, 253, 254, 255, 'others']

    # this variable is used to create a dictionary
    uni_hist_values = np.zeros(59)

    # initializing the upper and lower uniform histogram dictionaries
    upper_uni_hist = dict(zip(uni_hist_keys, uni_hist_values))
    lower_uni_hist = dict(zip(uni_hist_keys, uni_hist_values))

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

            # creating the upper and lower uniform histograms
            # if the value is uniform, increment its bin, otherwise incremente the 'others' bin
            if upper_pixel_value in uni_hist_keys:
                upper_uni_hist[upper_pixel_value] += 1
            else:
                upper_uni_hist['others'] += 1

            # the same operations for the lower histogram
            if lower_pixel_value in uni_hist_keys:
                lower_uni_hist[lower_pixel_value] += 1
            else:
                lower_uni_hist['others'] += 1

            # lower and upper pattern images, this bloc of code is used to display the two images
            # upper_pattern[i, j] = upper_pixel_value
            # lower_pattern[i, j] = lower_pixel_value

    # concatenating the upper and lower uniform histograms
    concatenated_hist = np.concatenate([list(upper_uni_hist.values()), list(lower_uni_hist.values())])

    return concatenated_hist

