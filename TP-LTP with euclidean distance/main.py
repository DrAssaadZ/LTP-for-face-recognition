'''
Zoubia Oussama, Zeghina Assaad Oussama
--------------------------------------
project description:
-------

the project is structured into 4 files (pre-processing, LTP, index_base and main) and 2 folders which contain the image
bases; the first folder is used for indexing which contains 360 images, and the second folder is used for testing which
contains 40 images.

-the pre-processing file contains 3 method (difference of gaussian, contrast equalization and the preprocessing)
the pre-processing method calls the other two methods and apply the pre-processing like in the paper, it is called in
the index_base file in the process of creating the index base and also interrogation

-the LTP file contains the LTP method which is used to apply LTP to a pre-processed image, this method is called in the
index_base file in the process of creating the index base and interrogation

-the index_base file contains two methods : create_index_base which is used to create an index base,
and the interrogate_index_base which queries the index base

-the main file contains offline phase code: which creates the base and Online phase code : which interrogates the base

-the euclidian distance is used as a similarity metric
'''


import index_base
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from os import path


# method used to display images
def show_images(images, cols=1, titles=None):

    # setting the titles of the images
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(0, n_images) ]
    titles[0] = 'Query'
    fig = plt.figure()
    # subplotting the images in the figure
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    # setting the size of the figure
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


# global parameters used for pre-processing and LTP
gamma = 0.2
alpha = 0.1
tau = 2.65
ltp_threshold = 0.2


# ================================================== OFFLINE PHASE =====================================================
# the path of the image base
image_base_path = './Faces'

# to run this code delete the index_base.npy file from the directory, it usually takes about 7 minutes
# checking if the index_base exists and creating it if it doesn't
if path.exists('index_base.npy'):
    print('The index file already exists in the directory')
else:
    # creating the index base
    try:
        print('Creating the index base please wait\n')
        index_base.create_index_base(path=image_base_path, gamma=gamma, alpha=alpha, tau=tau, ltp_threshold=ltp_threshold)
        print('\nIndex base successfully created\n')
    # displaying an error if it error occurred
    except NameError:
        print('Error occurred please check the entered path')


# ================================================== ONLINE PHASE ======================================================
# the path of the query image
query_img_path = './test faces/361.jpg'

# reading the query image as gray scale from the path and normalizing its pixel values between 0 and 1
img_query = cv.imread(query_img_path, cv.IMREAD_GRAYSCALE) / 255.0

# number of results to show
nbr_results = 9

# interrogating the index base by calling the interrogate_index_base function which returns a vector
# that contains [image names, distances to the index base images ]
query_result = index_base.interrogate_index_base(img=img_query, gamma=gamma, alpha=alpha, tau=tau,
                                                  ltp_threshold=ltp_threshold, nbr_result=nbr_results)

# creating a list that contains the corresponding images from the query result
img_array = [img_query]
for i in range(nbr_results):
    # reading the corresponding images from the query result as gray scale and normalizing it between 0 and 1
    img1 = cv.imread('./faces/' + query_result[i][0], cv.IMREAD_GRAYSCALE) / 255.0
    img_array.append(np.array(img1))

# displaying the corresponding images
show_images(img_array)
