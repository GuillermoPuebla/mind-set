from PIL import Image
from pathlib import Path

# read the checkerboard image
# img = Image.open(Path("assets") / "checkerboard.png")
img = Image.open(Path("assets") / "checkerboard_no_letters.png")
img = img.convert("L")

# # get the center pixel color
center_pixel = img.getpixel((img.width // 2, img.height // 2))  # (111, 111, 111, 255) / 111

print(center_pixel)

# turn all pixels with the same color to red
# for x in range(img.width):
#     for y in range(img.height):
#         if img.getpixel((x, y)) == center_pixel:
#             img.putpixel((x, y), (255, 0, 0, 255))

# img.show()

# to grey scale
# img_grey = img.convert("L")
# img_grey.show()

# # save the image
# img.save(Path("assets") / "checkerboard_grey.png")
