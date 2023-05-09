from PIL import Image, ImageDraw

# Triangle (Equilateral)
triangle = [(150, 50), (250, 250), (50, 250), (150, 50)]

# Square
square = [(100, 100), (200, 100), (200, 200), (100, 200), (100, 100)]

# Rectangle
rectangle = [(80, 100), (220, 100), (220, 200), (80, 200), (80, 100)]

# Hexagon (Regular)
hexagon = [
    (150, 50),
    (250, 100),
    (250, 200),
    (150, 250),
    (50, 200),
    (50, 100),
    (150, 50),
]

# Octagon (Regular)
octagon = [
    (150, 70),
    (210, 110),
    (210, 190),
    (150, 230),
    (90, 190),
    (90, 110),
    (150, 70),
]

# Pentagon (Regular)
pentagon = [(150, 50), (230, 120), (190, 230), (110, 230), (70, 120), (150, 50)]


def extend_line(line, factor):
    x1, y1, x2, y2 = line
    dx, dy = x2 - x1, y2 - y1
    x2_new, y2_new = x1 + dx * factor, y1 + dy * factor
    x1_new, y1_new = x1 - dx * (factor - 1), y1 - dy * (factor - 1)
    return (x1_new, y1_new, x2_new, y2_new)


# Define canvas dimensions
canvas_width, canvas_height = 300, 300

# Define the line coordinates
# pps = [(100, 50), (250, 80), (50, 250)]

# Create the original canvas and draw the line
# canvas_original = Image.new("RGB", (canvas_width, canvas_height), "black")
# draw_original = ImageDraw.Draw(canvas_original)
# draw_original.line(pps, fill="white", width=2)
# canvas_original.save("original_line.png")

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
        draw.line(points, fill="white", width=2)
    canvas_extended.show()


draw_shape(hexagon)
# Extend the line

# Draw the extended line
# draw_extended.line(extended_line, fill="white", width=2)
# canvas_extended.save("extended_line.png")
# canvas_extended.show()
