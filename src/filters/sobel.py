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

import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse

# To execute .\sobel.py -i IMAGEPATH

# Argument parser.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Image path or name if in the same folder")
args = vars(ap.parse_args())

# Read the image
image = cv2.imread(args["image"], 1)

# Check if not exist
if image is None:
    print("Image not found")
    exit(0)

# Convert to RGB colors
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convolute kernel = 5x5
sobelx = cv2.Sobel(image, cv2.CV_16S, 1, 0, ksize=3)
sobely = cv2.Sobel(image, cv2.CV_16S, 0, 1, ksize=3)

# Convert to uint8
abssobelx = cv2.convertScaleAbs(sobelx)
abssobely = cv2.convertScaleAbs(sobely)

# Gradient X & Y
grad = cv2.addWeighted(abssobelx, 0.5, abssobely, 0.5, 0)

# Original
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Imagen original')
plt.xticks([]), plt.yticks([])

# Roberts mask X & Y axis
plt.subplot(2, 2, 2)
plt.imshow(grad, cmap='gray')
plt.title('Sobel X & Y')
plt.xticks([]), plt.yticks([])

# Roberts mask X axis
plt.subplot(2, 2, 3)
plt.imshow(abssobelx, cmap='gray')
plt.title('Sobel X')
plt.xticks([]), plt.yticks([])

# Roberts mask Y axis
plt.subplot(2, 2, 4)
plt.imshow(abssobely, cmap='gray')
plt.title('Sobel Y')
plt.xticks([]), plt.yticks([])

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
