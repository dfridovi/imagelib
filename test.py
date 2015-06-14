"""
Test file to demonstrate imagelib functionality.
"""

import BasicFunctions as bf
from Sharpening import sharpen
from Blurring import blur
from FindEyes import findEyes

# import image
img = bf.imread("lotr.JPG")
#img = bf.imread("eye.png")
#img = bf.imread("obama.jpg")

# test blurring
#blurred = blur(img, mode="gaussian", k=5)
#bf.imshow(blurred)

# test sharpening
#sharpened = sharpen(img, k=21, lo_pass=True, min_diff=0.01, alpha=3.0)
#bf.imshow(sharpened)

# test eye detection
eyes = findEyes(img, mode="haar", train=None, mask=None)
print eyes
