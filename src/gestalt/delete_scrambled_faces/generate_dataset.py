import random
import glob
from PIL import Image, ImageOps, ImageFilter
import os
import pathlib
import shutil


face_folder = None
output_folder = None

face_folder = (
    pathlib.Path("assets") / "celebA_sample" / "normal"
    if face_folder is None
    else face_folder
)
output_folder = (
    pathlib.Path("data") / "gestalt" / "thatcher_illusion"
    if output_folder is None
    else output_folder
)

[shutil.rmtree(f) for f in glob.glob(str(output_folder / "**")) if os.path.exists(f)]
[
    (output_folder / cond).mkdir(parents=True, exist_ok=True)
    for cond in ["normal", "inverted", "thatcherized"]
]

f = glob.glob(str(face_folder / "**"))[1]


# Load the image
img = Image.open(f)
# Convert the image to grayscale
gray_img = img.convert("L")

# Invert the grayscale image to get a negative


negative_img = ImageOps.invert(gray_img)

# Apply a blur effect to the negative
blurred_img = negative_img.filter(ImageFilter.GaussianBlur(radius=5))

# Blend the grayscale image with the blurred negative
sketch_img = Image.blend(gray_img, blurred_img, alpha=0.6)
bw_img = sketch_img.convert("1", dither=Image.NONE)
bw_img.show()
assert False

image_facial_landmarks = np.array(get_image_facial_landmarks(f)).astype(int)
if (
    image_facial_landmarks == []
    or len(image_facial_landmarks) == 0
    or len(image_facial_landmarks) != 68
):
    print("NOPE!")

face_width = abs(image_facial_landmarks[16][0] - image_facial_landmarks[0][0])


def fix_rectangle(rectangle):
    rec = [tuple((i[1], i[0])) for i in rectangle]
    return [
        tuple([i - face_width * 0.06 for i in rec[0]]),
        tuple([i + face_width * 0.06 for i in rec[1]]),
    ]


left_eye_rectangle = fix_rectangle(
    get_bounding_rectangle(image_facial_landmarks[36:42])
)
right_eye_rectangle = fix_rectangle(
    get_bounding_rectangle(image_facial_landmarks[42:48])
)
mouth_rectangle = fix_rectangle(get_bounding_rectangle(image_facial_landmarks[48:68]))
face_width = abs(image_facial_landmarks[16][0] - image_facial_landmarks[0][0])

img = Image.open(f)


def extract_part(img, rectangle):
    return img.crop(
        (
            rectangle[0][0],
            rectangle[0][1],
            rectangle[1][0],
            rectangle[1][1],
        )
    )


# Extract eyes and mouth
left_eye = extract_part(img, left_eye_rectangle)
right_eye = extract_part(img, right_eye_rectangle)
mouth = extract_part(img, mouth_rectangle)
d = ImageDraw.Draw(img)

# Create a mask for the face region
mask = Image.new("L", img.size)
draw = ImageDraw.Draw(mask)
face_points = [(point[0], point[1]) for point in image_facial_landmarks]
draw.polygon(face_points, fill=255)

# Blur the entire image
blurred_img = img.filter(ImageFilter.BLUR)

# Combine the original and blurred images using the mask
final_img = Image.composite(img, blurred_img, mask)
final_img.show()
assert False

# d.rectangle(left_eye_rectangle)
# d.rectangle(right_eye_rectangle)
# d.rectangle(mouth_rectangle)
# img.show()

face_height = abs(image_facial_landmarks[8][1] - image_facial_landmarks[27][1])


# Function to place part of the image at a random position within face
def place_randomly_within_face(part, img):
    part_width, part_height = part.size
    face_start_x = image_facial_landmarks[0][0]
    face_start_y = image_facial_landmarks[27][1]
    rand_x = random.randint(face_start_x, face_start_x + face_width - part_width)
    rand_y = random.randint(face_start_y, face_start_y + face_height - part_height)
    img.paste(part, (rand_x, rand_y))


from PIL import ImageStat, ImageFilter, ImageOps

# Get the coordinates of the forehead
forehead_start = min([point[1] for point in image_facial_landmarks[19:25]])
forehead_end = max([point[1] for point in image_facial_landmarks[19:25]])

# Crop the forehead region
forehead = img.crop((0, forehead_start, face_width, forehead_end))

# Calculate the average color
mean_color = tuple(np.array(ImageStat.Stat(forehead).mean).astype(int))


def fill_rectangle(img, rectangle, color):
    # Create a new image filled with the desired color
    fill = Image.new(
        "RGB",
        (
            int(rectangle[1][0] - rectangle[0][0]),
            int(rectangle[1][1] - rectangle[0][1]),
        ),
        color=color,
    )

    # Paste the new image onto the original image
    img.paste(fill, (int(rectangle[0][0]), int(rectangle[0][1])))


# Change the color within the bounding boxes
fill_rectangle(img, left_eye_rectangle, mean_color)
fill_rectangle(img, right_eye_rectangle, mean_color)
fill_rectangle(img, mouth_rectangle, mean_color)
img.show()

# Randomly place eyes and mouth
# place_randomly_within_face(left_eye, img)
# place_randomly_within_face(right_eye, img)
# place_randomly_within_face(mouth, img)
# img.show()
