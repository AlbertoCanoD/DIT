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

# To execute .\roberts.py -i IMAGEPATH

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

# X axis Roberts mask
gx = np.array([[1, 0], [0, -1]])
robertx = cv2.filter2D(image, -1, gx)

# Y axis Roberts mask
gy = np.array([[0, 1], [-1, 0]])
roberty = cv2.filter2D(image, -1, gy)

# X & Y axis Roberts mask
imagex = cv2.filter2D(image, -1, gx)
imagey = cv2.filter2D(image, -1, gy)
robertxy = cv2.addWeighted(imagex, 0.5, imagey, 0.5, 0)

# Original
plt.subplot(2, 2, 1)
plt.imshow(image)
plt.title("Imagen original")
plt.xticks([]), plt.yticks([])

# Roberts mask X axis
plt.subplot(2, 2, 2)
plt.imshow(robertx)
plt.title('Filtro realce Roberts eje X')
plt.xticks([]), plt.yticks([])

# Roberts mask Y axis
plt.subplot(2, 2, 3)
plt.imshow(roberty)
plt.title('Filtro realce Roberts eje Y')
plt.xticks([]), plt.yticks([])

# Roberts mask X & Y axis
plt.subplot(2, 2, 4)
plt.imshow(robertxy)
plt.title('Filtro realce Roberts eje X e Y')
plt.xticks([]), plt.yticks([])

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
