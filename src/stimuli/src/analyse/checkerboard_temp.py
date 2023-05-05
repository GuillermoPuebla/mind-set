from PIL import Image
from pathlib import Path
import os
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image
from PIL import ImageDraw

coords_1 = [(950, 359), (1187, 410), (1058, 548), (819, 495)]
coords_2 = [(922, 694), (1170, 750), (1037, 896), (786, 839)]
# read image
img = Image.open(Path("assets") / "checkerboard_no_letters.png")


# the 4 coordinates form a polygon, get all pixel indices inside the polygon
def get_polygon_indices(coord_1, coord_2, coord_3, coord_4, img, name):
    mask = Image.new("L", img.size, 0)
    # draw a polygon on the mask
    ImageDraw.Draw(mask).polygon([coord_1, coord_2, coord_3, coord_4], outline=1, fill=255)
    mask = np.array(mask)
    # save the mask
    mask = Image.fromarray(mask)
    mask.save(Path("results", f"mask_{name}.png"))
    # get all pixel indices inside the polygon
    # indices = np.where(mask == 255)
    # return indices


indices_1 = get_polygon_indices(*coords_1, img, 1)
indices_2 = get_polygon_indices(*coords_2, img, 2)
