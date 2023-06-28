import numpy as np
from PIL import Image
from PIL import ImageDraw


def generate_grating(canvas_size, frequency, orientation, phase=0):
    width, height = canvas_size
    # Generate x and y coordinates
    x = np.linspace(-np.pi, np.pi, width)
    y = np.linspace(-np.pi, np.pi, height)
    x, y = np.meshgrid(x, y)

    # Rotate the grid by the specified orientation
    x_prime = x * np.cos(orientation) - y * np.sin(orientation)

    # Create the sinusoidal grating
    grating = 0.5 * (1 + np.sin(frequency * x_prime + phase))

    return grating


all_pil_images = []
freq = 10
canvas_size = (500, 500)
num_cycles = 1
num_points = 100

min_v = np.deg2rad(-20)
max_v = np.deg2rad(20)  # np.pi / 2
# Generate one cycle (up and down)
up = np.hstack((np.arange(min_v, max_v, 0.05), [max_v] * 0))
down = np.hstack((np.arange(max_v, min_v, -0.05), [min_v] * 0))

cycle = np.concatenate((up, down))

# Repeat the cycle
vector = np.tile(cycle, num_cycles)
for theta in cycle:
    print(theta)
    context = generate_grating(canvas_size, freq, theta)
    context = Image.fromarray(np.uint8(context * 255))
    # context.show()
    test = generate_grating(canvas_size, freq, 0)
    test = Image.fromarray(np.uint8(test * 255))

    mask = Image.new("L", test.size, 0)

    # Draw a white circle in the middle of the mask image
    draw = ImageDraw.Draw(mask)
    width, height = test.size
    radius = context.size[0] * 0.05
    center = (width // 2, height // 2)
    draw.ellipse(
        (
            center[0] - radius,
            center[1] - radius,
            center[0] + radius,
            center[1] + radius,
        ),
        fill=255,
    )

    # Paste the test image onto the context image using the mask
    context.paste(test, mask=mask)
    all_pil_images.append(context.convert("P"))
    # Save the resulting image
    # context.show()
all_pil_images[0].save(
    "animation.gif",
    save_all=True,
    append_images=all_pil_images[1:],
    optimize=True,
    duration=50,
    loop=0,
)
