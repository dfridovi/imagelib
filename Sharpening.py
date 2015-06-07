"""
Provides a function to sharpen an image.
"""

import numpy as np
from scipy.signal import convolve2d
import BasicFunctions as bf

def sharpen(img, edge_masking=False, k=11, var=1.0, max_difference=0.05,
            alpha=10.0, iter=3):
    """
    Sharpens the input image with unsharp masking. See Wikipedia
    (http://en.wikipedia.org/wiki/Unsharp_masking) for details. Note that
    this function only changes pixel values by a maximum of max_difference
    at each iteration.

    Provides optional edge masking functionality, where it attenuates 
    sharpening in the vicinity of edges.
    """
    
    # sharpen each color channel separately
    out = np.zeros(img.shape)
    for i in range(0, img.shape[2]):
        out[:, :, i] = convolve2d(img[:, :, i], mask, 
                                  mode="same", boundary="symm")
    
    return out
