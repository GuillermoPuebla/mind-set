from PIL import Image
from PIL.Image import new
from PIL.ImageDraw import Draw
from src.stimuli.src.utils.general import get_area_size


class ParentStimuli:
    def __init__(self, target_image_width=224, initial_expansion=4):
        # configs
        self.target_image_width = target_image_width
        self.initial_expansion = initial_expansion  # alias, larger the more aliasing

        # create canvas
        self.initial_image_width = self.target_image_width * self.initial_expansion

        self.canvas = new("RGBA", (self.initial_image_width,) * 2, color=(0, 0, 0, 0))
        self.draw = Draw(self.canvas)

        # contained shape
        self.contained_shapes = []

    # helpers -----------------------------------------------------------------
    def count_pixels(self, color):
        return get_area_size(self.canvas, color)

    def _add_background(self):
        new_canvas = new("RGBA", (self.initial_image_width,) * 2, color=(0, 0, 0, 255))
        new_canvas.paste(self.canvas, (0, 0), self.canvas)
        self.canvas = new_canvas
        return self

    def _shrink(self):
        self.canvas = self.canvas.resize(
            (self.target_image_width, self.target_image_width),
            resample=Image.Resampling.LANCZOS,
        )
        return self

    def _center_shapes(self):
        cropped_canvas = self.canvas.crop(self.canvas.getbbox())

        new_canvas = new("RGBA", (self.initial_image_width,) * 2, color=(0, 0, 0, 255))
        new_canvas.paste(
            cropped_canvas,
            (
                int((self.initial_image_width - cropped_canvas.width) / 2),
                int((self.initial_image_width - cropped_canvas.height) / 2),
            ),
            cropped_canvas,
        )
        self.canvas = new_canvas
        return self

    def _binary_filter(self):
        # make all the non 0 pixels 255
        self.canvas = self.canvas.point(lambda p: p > 0 and 255)
        return self

    def rotate(self, degrees: int, keep_within_canvas: bool = True):
        bbox = self.canvas.getbbox()
        cropped = self.canvas.crop(bbox)
        rotated = cropped.rotate(degrees, expand=True)
        dx, dy = rotated.size
        new_canvas = Image.new("RGBA", (self.initial_image_width,) * 2, (0, 0, 0, 0))

        # Calculate the center point of the cropped image
        center_x, center_y = (bbox[2] + bbox[0]) // 2, (bbox[3] + bbox[1]) // 2

        if keep_within_canvas:
            # Calculate the new bounding box based on the center point
            new_left = max(0, min(center_x - dx // 2, self.initial_image_width - dx))
            new_upper = max(0, min(center_y - dy // 2, self.initial_image_width - dy))
        else:
            new_left = center_x - dx // 2
            new_upper = center_y - dy // 2

        new_bbox = (new_left, new_upper, new_left + dx, new_upper + dy)
        new_canvas.paste(rotated, new_bbox)
        self.canvas = new_canvas
        self.rotation = degrees
        return self

    def move_to(self, coordinates: tuple, keep_within_canvas: bool = True):
        """
        crop the image to the bounding box of the shape and move it to the specified coordinates
        1. crop the image to the bounding box of the shape
        2. create a new image with transparent background
        3. calculate the new bounding box of the shape
        4. if the new bounding box is outside the image, move the shape to the closest edge
        5. paste the shape on the new image
        """
        x, y = map(lambda v: int(v * self.initial_image_width), coordinates)
        bbox = self.canvas.getbbox()
        cropped = self.canvas.crop(bbox)
        new_bbox = (x - cropped.width // 2, y - cropped.height // 2)
        new_canvas = Image.new("RGBA", (self.initial_image_width,) * 2, (0, 0, 0, 0))
        paste_box = (
            new_bbox[0],
            new_bbox[1],
            new_bbox[0] + cropped.width,
            new_bbox[1] + cropped.height,
        )

        if keep_within_canvas:
            paste_pos = (
                min(max(paste_box[0], 0), self.initial_image_width - cropped.width),
                min(max(paste_box[1], 0), self.initial_image_width - cropped.height),
            )
        else:
            paste_pos = (paste_box[0], paste_box[1])

        new_canvas.paste(cropped, paste_pos)
        self.canvas = new_canvas
        return self
