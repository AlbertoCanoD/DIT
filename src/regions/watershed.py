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

# To execute .\watershed.py -i IMAGEPATH

# Argument parser.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Image path or name if in the same folder")
args = vars(ap.parse_args())

# Read the image
image = cv2.imread(args["image"])

# Copy of the original image
imageAux = image

# Check if not exist
if image is None:
    print("Image not found")
    exit(0)

# Convert the image and imageAux to RGB, and make a gray copy of image
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
imageAux = cv2.cvtColor(imageAux, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Make an Otsu thresholding
ret, thresh = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Remove noise
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Find background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Find object area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

# Find edge area
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Mark elements
ret, markers = cv2.connectedComponents(sure_fg)

# Make the background = 1
markers = markers+1

# Mark unknow area with 0
markers[unknown == 255] = 0

# Mark boundary region with -1
markers = cv2.watershed(image, markers)
image[markers == -1] = [255, 0, 0]

# Original
plt.subplot(241)
plt.imshow(imageAux)
plt.title("Imagen original")
plt.xticks([]), plt.yticks([])

# Thresholding of image
plt.subplot(242)
plt.imshow(thresh, 'gray')
plt.title('Umbralizado')
plt.xticks([]), plt.yticks([])

# Noise reduction
plt.subplot(243)
plt.imshow(opening, 'gray')
plt.title('Eliminacion del ruido')
plt.xticks([]), plt.yticks([])

# Background
plt.subplot(244)
plt.imshow(sure_bg)
plt.title('Fondo de la imagen')
plt.xticks([]), plt.yticks([])

# Foreground
plt.subplot(245)
plt.imshow(sure_fg)
plt.title('Objetos de la imagen')
plt.xticks([]), plt.yticks([])

# Edges
plt.subplot(246)
plt.imshow(unknown)
plt.title('Bordes')
plt.xticks([]), plt.yticks([])

# Image marked
plt.subplot(247)
plt.imshow(markers)
plt.title('Imagen etiquetada')
plt.xticks([]), plt.yticks([])

# Image processed
plt.subplot(248)
plt.imshow(image)
plt.title("Imagen segmentada")
plt.xticks([]), plt.yticks([])

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
