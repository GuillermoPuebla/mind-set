from PIL import Image, ImageDraw

# Every polygon is first describe by a list of points in a canvas that is 100x100. The points will be then translated and rescaled depending on the canvas size and on the "obj_to_canvas" parameter.

# Square
square = ((0, 0), (100, 0), (100, 100), (0, 100), (0, 0))


def calculate_centroid(points):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    centroid_x = (max(x_coords) + min(x_coords)) / 2
    centroid_y = (max(y_coords) + min(y_coords)) / 2
    return centroid_x, centroid_y


def translate_to_center(points, canvas_width, canvas_height):
    centroid = calculate_centroid(points)
    translation_x = canvas_width / 2 - centroid[0]
    translation_y = canvas_height / 2 - centroid[1]

    translated_points = [(p[0] + translation_x, p[1] + translation_y) for p in points]
    return translated_points


def extend_line(line, factor):
    x1, y1, x2, y2 = line
    dx, dy = x2 - x1, y2 - y1
    x2_new, y2_new = x1 + dx * factor, y1 + dy * factor
    x1_new, y1_new = x1 - dx * (factor - 1), y1 - dy * (factor - 1)
    return (x1_new, y1_new, x2_new, y2_new)


# Define canvas dimensions
canvas_width, canvas_height = 200, 200
# Create the extended canvas and draw the extended line
canvas_extended = Image.new("RGB", (canvas_width, canvas_height), "black")
draw = ImageDraw.Draw(canvas_extended)
extend = False


def draw_shape(points):
    for i in range(len(points) - 1):
        if extend:
            line = extend_line(points[i] + points[i + 1], 10)
        else:
            line = points[i] + points[i + 1]
        draw.line(line, fill="white", width=2)
    canvas_extended.show()


square_centered = translate_to_center(square, canvas_width, canvas_height)
draw_shape(square_centered)
# Extend the line

# Draw the extended line
# draw_extended.line(extended_line, fill="white", width=2)
# canvas_extended.save("extended_line.png")
# canvas_extended.show()
