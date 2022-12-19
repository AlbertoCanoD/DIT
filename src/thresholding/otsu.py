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

# To execute .\otsu.py -i1 IMAGEPATH -i2 IMAGEPATH

# Argument parser.
ap = argparse.ArgumentParser()
ap.add_argument("-i1", "--image1", required=True,
                help="Image path or name if in the same folder")

ap.add_argument("-i2", "--image2", required=True,
                help="Image path or name if in the same folder")

args = vars(ap.parse_args())

# Read the image1
image1 = cv2.imread(args["image1"], 0)

# Check if not exist
if image1 is None:
    print("Image not found")
    exit(0)

# Read the image2
image2 = cv2.imread(args["image2"], 0)

# Check if not exist
if image2 is None:
    print("Image not found")
    exit(0)

ret1, otsu1 = cv2.threshold(image1, 0, 255,
                            cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret2, otsu2 = cv2.threshold(image2, 0, 255,
                            cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Image 1
plt.subplot(2, 2, 1)
plt.imshow(image1,  'gray')
plt.title("Imagen 1")
plt.xticks([]), plt.yticks([])

# THRESH_OTSU image 1
plt.subplot(2, 2, 2)
plt.imshow(otsu1,  'gray')
plt.title('THRESH_OTSU T=' + str(int(ret1)))
plt.xticks([]), plt.yticks([])

# Image 2
plt.subplot(2, 2, 3)
plt.imshow(image2, 'gray')
plt.title('Imagen 2')
plt.xticks([]), plt.yticks([])

# # THRESH_OTSU image 2
plt.subplot(2, 2, 4)
plt.imshow(otsu2, 'gray')
plt.title('THRESH_OTSU T=' + str(int(ret2)))
plt.xticks([]), plt.yticks([])

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
