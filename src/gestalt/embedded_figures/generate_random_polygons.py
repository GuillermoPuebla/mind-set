import random

import numpy as np
from PIL import Image, ImageDraw

"""
Note: these functions do not guarantee that the polygon will be simple (e.g. without intersection). Also the angles will sometime be similar to each other so that it's difficult to see whether there are really the desired number of vertices. The way to use it is to generate several polygons and inspect them to find those that satisfy your requirements. Then copy the vertices and use them in other scripts (e.g. in generate_dataset.py in the embedded_figure dataset)
"""


def generate_random_polygon(n_vertices, canvas_size=100):
    # List to store the vertices
    vertices = []
    eps = 0.1
    # Generate n_vertices random angles
    # Generate the first angle
    angle = random.uniform(0, tau)

    # Generate the list of angles, each at least eps away from the previous one
    angles = [angle]
    for _ in range(1, n_vertices):
        angle = (angle + eps + random.uniform(0, tau - eps)) % tau
        angles.append(angle)
    angles.sort()
    print(angles)
    # Define a minimum radius to ensure that the polygon covers more area
    # min_radius = canvas_size / 2 - canvas_size / 10

    # For each angle, generate a random point on the circle with radius between min_radius and canvas_size/2
    for angle in angles:
        radius = random.uniform(0, canvas_size / 2)
        x = canvas_size / 2 + radius * cos(angle)
        y = canvas_size / 2 + radius * sin(angle)
        vertices.append((x, y))

    # Add the first vertex at the end to close the polygon
    vertices.append(vertices[0])

    # Return the vertices
    return vertices


def change_range(x, initial_range, finale_range):
    # Calculate the scale factor between the initial range and the finale range
    scale_factor = (finale_range[1] - finale_range[0]) / (
        initial_range[1] - initial_range[0]
    )

    # Apply the scale factor and shift to the number
    transformed_x = (x - initial_range[0]) * scale_factor + finale_range[0]

    return transformed_x


##

from math import cos, sin, tau
import PIL.Image as Image

img = Image.new("RGB", (100, 100), "white")
draw = ImageDraw.Draw(img)

# either use .polygon(), if you want to fill the area with a solid colour
vertices = generate_random_polygon(7)
rescaled_vertices = [
    tuple(change_range(x, [np.min(vertices), np.max(vertices)], [0, 100]))
    for x in vertices
]
draw.line(rescaled_vertices, width=1, fill="black")
img.show()

##
