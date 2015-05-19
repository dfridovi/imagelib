"""
Provides a function to blur an image.
"""

import numpy as np
from scipy.signal import convolve2d
import BasicFunctions as bf

def blur(img, mode="gaussian", k=11, var=1.0):
    """
    Blurs image with a kernel mask of the given type. Supports the following
    modes, each of which can have varying size k:
      (1) gaussian: can also provide variance var
      (2) box: no additional parameters needed
    """
    
    if mode == "gaussian":
        mask = bf.gauss_mask(k, var)
        
    elif mode == "box":
        mask = bf.box_mask(k)
        
    else: 
        raise Exception("Mode %s not supported." % mode)

    return convolve2d(img, mask, mode="same", boundary="symm")
