"""
A set of basic functions for use in this library.
"""

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

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

def gauss_mask(k, var=1.0):
    """ Return k-by-k 2D Gaussian on the range [-3sd, +3sd] in both x,y."""

    xvals = np.linspace(-3.0*np.sqrt(var), 3.0*np.sqrt(var), k)
    yvals = np.linspace(3.0*np.sqrt(var), -3.0*np.sqrt(var), k)
    
    mask = np.zeros((k, k), dtype=np.float)
    for i in range(k):
        for j in range(k):
            mask[i, j] = np.exp(-0.5 * (xvals[j]**2 + yvals[i]**2) / var)
            
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
