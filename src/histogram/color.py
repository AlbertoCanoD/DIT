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

# To execute .\color.py -i IMAGEPATH

# Argument parser.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Image path or name if in the same folder")
args = vars(ap.parse_args())

# Read the image
image = cv2.imread(args["image"])

# Check if not exist
if image is None:
    print("Image not found")
    exit(0)

# Convert to RGB colors
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax1 = plt.subplot(221)
ax2 = plt.subplot(223)

# Image
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Imagen')
ax1.axis("off")

# RGB Histogram of image
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv2.calcHist([image], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])
    plt.xlabel("Niveles de color")
    plt.ylabel("Frecuencia relativa")
    ax2.set_title('Histograma RGB')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
