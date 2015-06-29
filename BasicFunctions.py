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

    return (int(ctr[0] - round(0.5*shape[0])), int(ctr[1] - round(0.5*shape[1])))

def tl2center(tl, shape):
    """ Convert top left corner of box to center. """

    return (int(tl[0] + round(0.5*shape[0])), int(tl[1] + round(0.5*shape[1])))

def px2cell(px, cell_shape=(8, 8)):
    """ Convert pixel index to cell index. """

    return (int(px[0] / cell_shape[0]), int(px[1] / cell_shape[1]))

def cell2px(cell, cell_shape=(8, 8)):
    """ Convert cell index to pixel index. """

    return (int(cell[0] * cell_shape[0]), int(cell[1] * cell_shape[1]))

def drawRectangle(img, center, shape, color):
    """ Draw a rectangle on the image. """

    tl = center2tl(center, shape)
    img[tl[0]:tl[0]+shape[0], tl[1]:tl[1]+1, :] = color
    img[tl[0]:tl[0]+shape[0], tl[1]+shape[1]:tl[1]+shape[1]+1, :] = color
    img[tl[0]:tl[0]+1, tl[1]:tl[1]+shape[1], :] = color
    img[tl[0]+shape[0]:tl[0]+shape[0]+1, tl[1]:tl[1]+shape[1], :] = color

def flat2ind(n, ncol):
    i = n / ncol
    j = n % ncol
    return (i, j)

def getHog(img, orientations=9, cell_shape=(8, 8), normalize=True, flatten=True):
    """ 
    Compute histogram of oriented gradients for the given grayscale image. 
    Default is to do image-level normalization, and return a flattened 1D 
    feature vector. Can also return 3D array of histograms indexed by cell 
    position, which can be normalized and flattened on the fly for 
    better performance.
    """

    # compute gradient magnitude and orientation for entire image
    grad_x = convolve2d(img, sobelX_mask(), mode="same", boundary="symm")
    grad_y = convolve2d(img, sobelY_mask(), mode="same", boundary="symm")
    grad_mag = np.sqrt(np.multiply(grad_x, grad_x) + 
                       np.multiply(grad_y, grad_y))
    grad_orient = np.arctan2(grad_y, grad_x)

    # now, cell-by-cell, compute histograms
    num_cells_x = img.shape[1] / cell_shape[1]
    num_cells_y = img.shape[0] / cell_shape[0]

    hists_out = np.empty((num_cells_y, num_cells_x, orientations), 
                         dtype=np.float)
    for i in range(0, num_cells_y):
        for j in range(0, num_cells_x):
            cell_orient = grad_orient[i*cell_shape[0]:(i+1)*cell_shape[0],
                                      j*cell_shape[1]:(j+1)*cell_shape[1]]
            cell_mag = grad_mag[i*cell_shape[0]:(i+1)*cell_shape[0],
                                j*cell_shape[1]:(j+1)*cell_shape[1]]
            hist, bins = np.histogram(cell_orient, bins=orientations, 
                                      weights=cell_mag)
            hists_out[i, j, :] = hist

    # normalize if needed
    if normalize:
        hists_out = normalizeHog(hists_out)

    # flatten if needed
    if flatten:
        return hists_out.ravel()

    return hists_out

def normalizeHog(hog):
    """ Divide each element of HOG by the total energy. """

    return hog / (np.sqrt(hog.sum()**2 + 1.0e-5)
