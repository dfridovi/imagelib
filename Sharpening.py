"""
Provides a function to sharpen an image.
"""

import numpy as np
from scipy.signal import convolve2d
import BasicFunctions as bf

def sharpen(img, k=11, lo_pass=True, min_diff=0.05, alpha=1.0):
    """
    Sharpens the input image with unsharp masking. See Wikipedia
    (http://en.wikipedia.org/wiki/Unsharp_masking) for details. Note that
    this function only changes pixel values by a maximum of max_difference
    at each iteration.

    Optionally applies a low-pass Gaussian filter at the end, 
    to reduce the effect of high frequency noise.
    """
    
    # sharpen each color channel separately
    out = np.zeros(img.shape)
    sharp_mask = bf.gauss_mask(k)
    for i in range(0, img.shape[2]):
        blurred = convolve2d(img[:, :, i], sharp_mask, 
                             mode="same", boundary="symm")
        diff = img[:, :, i] - blurred
        diff[np.abs(diff) < min_diff] = 0.0
        scaled = alpha * diff
        out[:, :, i] = img[:, :, i] + scaled

    # truncate to [0, 1]
    out = bf.truncate(out)        
    if lo_pass:

        # if necessary, blur each color channel separately
        blur_mask = bf.gauss_mask(2 * int(1.0 + (k - 1) / 8.0) + 1) 
        for i in range(0, img.shape[2]):
            out[:, :, i] = convolve2d(img[:, :, i], blur_mask, 
                                      mode="same", boundary="symm")

    return out
