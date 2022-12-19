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

# To execute .\ecualizacion.py -i IMAGEPATH

# Argument parser.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Image path or name if in the same folder")
args = vars(ap.parse_args())

# Read the image and convert to grayscale
image = cv2.imread(args["image"], cv2.IMREAD_GRAYSCALE)

# Check if not exist
if image is None:
    print("Image not found")
    exit(0)

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax1 = plt.subplot(221)
ax2 = plt.subplot(222)
ax3 = plt.subplot(223)
ax4 = plt.subplot(224)

# Original Image
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Imagen original')
ax1.axis("off")

# Equalized Image
imageEq = cv2.equalizeHist(image)
ax2.set_title('Imagen ecualizada')
ax2.imshow(imageEq, cmap=plt.cm.gray)
ax2.axis("off")

# Histogram of image
ax3.set_title('Histograma')
plt.xlabel("Niveles de gris")
plt.ylabel("Frecuencia relativa")
ax3.hist(image.ravel(), 256)

# Histogram of equalized image
ax4.set_title('Ecualización del histograma')
plt.xlabel("Niveles de gris")
plt.ylabel("Frecuencia relativa")
ax4.hist(imageEq.ravel(), 256)

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
