"""
Test file to demonstrate imagelib functionality.
"""

import BasicFunctions as bf
from Sharpening import sharpen
from Blurring import blur

# import image
img = bf.imread("eye.png")

# test blurring
#blurred = blur(img, mode="gaussian", k=5)
#bf.imshow(blurred)

# test sharpening
sharpened = sharpen(img, k=21, lo_pass=True, min_diff=0.01, alpha=5.0)
bf.imshow(sharpened)
