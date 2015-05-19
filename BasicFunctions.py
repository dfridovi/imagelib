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

def imshow(img, title="", cmap="", cbar=False):
    """ Show image to screen. """
    
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    
    if cbar:
        plt.colorbar()

def imsave(img, imfile):
    """ Save image to file."""

    mpimg.imsave(imfile, img)
