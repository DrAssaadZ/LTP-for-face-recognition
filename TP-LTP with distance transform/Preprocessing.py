import cv2
import numpy as np


# a method that applies the difference of gaussian to an input image
def difference_of_gaussian(img, sigma1=1, sigma2=2):
    # difference of gaussian
    img_blur1 = cv2.GaussianBlur(img, (0, 0), sigma1, borderType=cv2.BORDER_REPLICATE)
    img_blur2 = cv2.GaussianBlur(img, (0, 0), sigma2, borderType=cv2.BORDER_REPLICATE)
    img_dog = (img_blur1 - img_blur2)

    # normalize the pixel values of the DoG images between -1 and 1
    img_dog = img_dog / np.max(np.abs(img_dog))

    return img_dog


# a method that applies the contrast equalization to an input image
def contrast_equalization(img, alpha, tau):
    # Contrast equalisation
    # contrast equalization equation 1
    img_contrast1 = np.abs(img)
    img_contrast1 = np.power(img_contrast1, alpha)
    img_contrast1 = np.mean(img_contrast1)
    img_contrast1 = np.power(img_contrast1, 1.0 / alpha)
    img_contrast1 = img / img_contrast1

    # contrast equalization equation 2
    img_contrast2 = np.abs(img_contrast1)
    img_contrast2 = img_contrast2.clip(0, tau)
    img_contrast2 = np.mean(img_contrast2)
    img_contrast2 = np.power(img_contrast2, 1.0 / alpha)
    img_contrast2 = img_contrast1 / img_contrast2
    img_contrast2 = tau * np.tanh((img_contrast2 / tau))

    # rescale the pixel values between 0 and 255
    img_contrast2 = (255.0 * (0.5 * img_contrast2 + 0.5)).clip(0, 255).astype(np.uint8)

    return img_contrast2


# a method that contains the three main steps of the pre-processing discussed in the paper ( gamma correction, DoG,
# contrast equalisation(CE)), it takes three parameters( gamma value for the gamma correction, alpha and tau for the CE)
# the function returns the pre-processed image
def pre_processing(img, gamma=0.2, alpha=0.10, tau=2.65):

    # gamma correction
    img_gamma = np.power(img, gamma)

    # difference of gaussian
    img_dog = difference_of_gaussian(img_gamma, )

    # contrast equalization
    img_contrast = contrast_equalization(img_dog, alpha, tau)

    return img_contrast


