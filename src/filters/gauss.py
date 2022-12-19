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

# To execute .\gauss.py -i IMAGEPATH -k KERNELSIZE

# Argument parser.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Image path or name if in the same folder")

ap.add_argument("-k", "--kernel", type=int, help="Kernel size, > 3")

ap.add_argument("-s", "--sigma", type=float, help="Sigma value")

args = vars(ap.parse_args())

# Read the image
image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Check if not exist
if image is None:
    print("Image not found")
    exit(0)

# Processing kernel size
ksize = args["kernel"]

if ksize is None or ksize < 3:
    ksize = 3

kernel = (ksize, ksize)

# Processing sigma size
sigma = args["sigma"]

if sigma is None:
    sigma = 0

# Apply the gaussian blur with kernel and sigma
blur = cv2.GaussianBlur(image, kernel, sigma)

# Original
plt.subplot(121)
plt.imshow(image)
plt.title("Imagen original")
plt.xticks([]), plt.yticks([])

# Blurred image
plt.subplot(122)
plt.imshow(blur)
plt.title('Filtro gaussiano con kernel ' + str(ksize) + 'x' +
          str(ksize) + ' y desviacion tipica ' + str(sigma))
plt.xticks([]), plt.yticks([])

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
