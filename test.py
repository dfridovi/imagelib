"""
Test file to demonstrate imagelib functionality.
"""

import BasicFunctions as bf
from Sharpening import sharpen
from Blurring import blur
from FindEyes import findEyes, tl2center

# import image
#img = bf.imread("lotr.JPG")
#img = bf.imread("eye.png")
img = bf.imread("obama.jpg")

# test blurring
#blurred = blur(img, mode="gaussian", k=5)
#bf.imshow(blurred)

# test sharpening
#sharpened = sharpen(img, k=21, lo_pass=True, min_diff=0.01, alpha=3.0)
#bf.imshow(sharpened)

# test eye detection
eye_shape = (25, 50)
eye1_tl = (200, 480)
eye2_tl = (195, 655)
eye1_ctr = tl2center(eye1_tl, eye_shape)
eye2_ctr = tl2center(eye2_tl, eye_shape)
eyes = findEyes(img, mode="svm", train=img, eye_centers=[eye1_ctr, eye2_ctr], 
                eye_shape=eye_shape, svm=None, scaler=None, 
                locs=[eye1_tl, eye2_tl])
print eyes
