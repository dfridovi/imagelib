# imagelib
An image processing library for recreational and scientific use.

## Philosophy
This library is intended for use by recreational photographers, but also by researchers. I have attempted to satisfy both communities: on the one hand, this library attempts to provide useful, user-friendly, fast tools for the practitioner, but on the other hand, it is also written from the ground up, and does not rely extensively on other related implementations. I hope that it is useful, complete, and above all, well-documented.

## General organization
Each code file in this library represents one of the following.

1. A data type used by other files.
2. A set of basic functions (such as loading an image) which will be called by many of the other functions in the library.
3. A function intended to be called by a main() function in another file. These functions implement different procedures, for example image sharpening, or blurring, or object detection, or focus detection.

## Functionality
The following functions are currently supported.
* Variable image blurring
* Variable image sharpening, with optional noise reduction
* Eye detection
* Eye tracking in video

## In the works...
The following functionality will be added soon!
* Pupil detection
* Gaze estimation
* Deconvolution (blind/non-blind)
* Focus detection