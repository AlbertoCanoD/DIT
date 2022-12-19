'''TrabajoTDI
    Copyright (C) 2022  Alberto Cano Delgado
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.'''

import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

# To execute .\simple.py -i IMAGEPATH -t THRESHOLD

# Argument parser.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Image path or name if in the same folder")

ap.add_argument("-t", "--threshold", help="Threshold value")

args = vars(ap.parse_args())

# Read the image
image = cv2.imread(args["image"], 0)

# Check if not exist
if image is None:
    print("Image not found")
    exit(0)

threshold = args["threshold"]

if threshold is None or threshold < 0 or threshold > 255:
    threshold = 127

# Types of thresholding
ret, bin = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
ret, binInv = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
ret, trunc = cv2.threshold(image, threshold, 255, cv2.THRESH_TRUNC)
ret, tozero = cv2.threshold(image, threshold, 255, cv2.THRESH_TOZERO)
ret, tozeroInv = cv2.threshold(image, threshold, 255, cv2.THRESH_TOZERO_INV)

# Original
plt.subplot(2, 3, 1)
plt.imshow(image, 'gray')
plt.title("Imagen original")
plt.xticks([]), plt.yticks([])

# THRESH_BINARY
plt.subplot(2, 3, 2)
plt.imshow(bin, 'gray')
plt.title('THRESH_BINARY')
plt.xticks([]), plt.yticks([])

# THRESH_BINARY_INV
plt.subplot(2, 3, 3)
plt.imshow(binInv, 'gray')
plt.title('THRESH_BINARY_INV')
plt.xticks([]), plt.yticks([])

# THRESH_TRUNC
plt.subplot(2, 3, 4)
plt.imshow(trunc, 'gray')
plt.title('THRESH_TRUNC')
plt.xticks([]), plt.yticks([])

# THRESH_TOZERO
plt.subplot(2, 3, 5)
plt.imshow(tozero, 'gray')
plt.title('THRESH_TOZERO')
plt.xticks([]), plt.yticks([])

# THRESH_TOZERO_INV
plt.subplot(2, 3, 6)
plt.imshow(tozeroInv, 'gray')
plt.title('THRESH_TOZERO_INV')
plt.xticks([]), plt.yticks([])

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
