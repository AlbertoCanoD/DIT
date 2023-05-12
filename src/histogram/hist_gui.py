# import the necessary packages
import tkinter as tk
from tkinter.filedialog import askopenfilename
from PIL import Image
from PIL import ImageTk
import cv2
import imutils
from gray import gray_hist


# initialize the window toolkit along with the two image panels
root = tk.Tk()

def select_image():
    # grab a reference to the image panels
    global panelA, panelB
    # open a file chooser dialog and allow the user to select an input
    # image
    path = askopenfilename()
    # ensure a file path was selected
    if len(path) > 0:
        # load the image from disk, convert it to grayscale, and detect
        # edges in it
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 50, 100)
        # OpenCV represents images in BGR order; however PIL represents
        # images in RGB order, so we need to swap the channels
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # convert the images to PIL format...
        image = Image.fromarray(image)
        edged = Image.fromarray(edged)
        # ...and then to ImageTk format
        image = ImageTk.PhotoImage(image)
        edged = ImageTk.PhotoImage(edged)
        # if the panels are None, initialize them
        if panelA is None or panelB is None:
            # the first panel will store our original image
            panelA = tk.Label(image=image)
            panelA.image = image
            panelA.pack(side="left", padx=10, pady=10)
            # while the second panel will store the edge map
            panelB = tk.Label(image=edged)
            panelB.image = edged
            panelB.pack(side="right", padx=10, pady=10)
        # otherwise, update the image panels
        else:
            # update the pannels
            panelA.configure(image=image)
            panelB.configure(image=edged)
            panelA.image = image
            panelB.image = edged


def hist_options():
    path = askopenfilename()
    option = tk.IntVar()

rad1 = tk.Radiobutton(root, text='GRAY', width=25, value=1, variable=option, command=gray_hist(path))
'''rad2 = tk.Radiobutton(root, text='Amarillo', width=25, value=2,
                      variable=option, command=deteccion_color)
rad3 = tk.Radiobutton(root, text='Azul celeste', width=25,
                      value=3, variable=option, command=deteccion_color)'''
rad1.grid(column=0, row=4)
'''rad2.grid(column=0, row=5)
rad3.grid(column=0, row=6)
'''

panelA = None
panelB = None
# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI


btn = tk.Button(root, text="Select an image", command=select_image)

# Select the image mode
hist_mode = tk.Button(root, text="Select the hist mode", command=hist_options)


btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
# kick off the GUI
root.mainloop()


# https://analyticsindiamag.com/real-time-gui-interactions-with-opencv-in-python/
# https://www.educba.com/opencv-gui/
# https://omes-va.com/tkinter-opencv-imagen/
