"""
A set of basic functions for use in this library.
"""

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def imread(imfile):
    """ Read image from file and normalize. """

    img = mpimg.imread(imfile).astype(np.float)
    img = (img - img.min()) / (img.max() - img.min())
    return img

def imshow(img, title="", cmap="gray", cbar=False):
    """ Show image to screen. """
    
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    
    if cbar:
        plt.colorbar()

    plt.show()

def imsave(img, imfile):
    """ Save image to file."""

    mpimg.imsave(imfile, img)

def gauss_mask(k):
    """ 
    Return k-by-k 2D Gaussian on the range [-3, +3] in both x,y.
    Assumes unit variance (as a normalization). 
    """

    xvals = np.linspace(-3.0, 3.0, k)
    yvals = np.linspace(3.0, -3.0, k)
    
    mask = np.zeros((k, k), dtype=np.float)
    for i in range(k):
        for j in range(k):
            mask[i, j] = np.exp(-0.5 * (xvals[j]**2 + yvals[i]**2))
            
    mask = mask / mask.sum()
    return mask

def sobelX_mask():
    """ Return Sobel x-direction mask."""

    return 0.125 * np.array([[-1, 0, 1],
                             [-2, 0, 1],
                             [-1, 0, 1]])

def sobelY_mask():
    """ Return Sobel y-direction mask."""

    return 0.125 * np.array([[-1, -2, -1],
                             [0, 0, 0],
                             [1, 2, 1]])

def box_mask(k):
    """ Return a k-by-k box filter mask."""

    return (1.0 / (k * k)) * np.ones((k, k))

def truncate(img):
    """ Truncate values in image to range [0.0, 1.0]. """

    img[img > 1.0] = 1.0
    img[img < 0.0] = 0.0
    return img

def rgb2gray(img):
    """ Convert an RGB image to grayscale. """

    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    return 0.299*r + 0.587*g + 0.114*b

def center2tl(ctr, shape):
    """ Convert center of box to top left corner. """

    return (ctr[0] - round(0.5*shape[0]), ctr[1] - round(0.5*shape[1]))

def tl2center(tl, shape):
    """ Convert top left corner of box to center. """

    return (tl[0] + round(0.5*shape[0]), tl[1] + round(0.5*shape[1]))

def drawRectangle(img, center, shape, color):
    """ Draw a rectangle on the image. """

    tl = center2tl(center, shape)
    img[tl[0]:tl[0]+shape[0], tl[1]:tl[1]+1, :] = color
    img[tl[0]:tl[0]+shape[0], tl[1]+shape[1]:tl[1]+shape[1]+1, :] = color
    img[tl[0]:tl[0]+1, tl[1]:tl[1]+shape[1], :] = color
    img[tl[0]+shape[0]:tl[0]+shape[0]+1, tl[1]:tl[1]+shape[1], :] = color

def hog(img, orientations=9, pix_per_cell=(8, 8), 
        cells_per_block=(3, 3), normalise=True):
    """ 
    Compute histogram of oriented gradients for the given image. Default is to do
    block-level normalization 
    """
