from PIL.Image import new
from PIL.ImageDraw import Draw
from src.stimuli.src.utils.color import get_rgba_color
from PIL import Image
import numpy as np
import pprint
from src.stimuli.src.utils.general import get_area_size
from matplotlib import colors
from src.stimuli.src.utils.shapes import SizeMonitor
import math


class ShapeCoreFunctions:
    def __init__(self, parent, init_img=None):
        self.index = len(parent.contained_shapes)
        self.color = get_rgba_color(self.index)

        parent.contained_shapes.append(self)

        self.parent = parent
        self.initial_image_width = self.parent.initial_image_width

        if init_img:  # secondary shapes
            self.canvas = init_img
            self.set_color()
        else:
            self.canvas = new(
                "RGBA", (self.initial_image_width,) * 2, color=(0, 0, 0, 0)
            )

        self.draw = Draw(self.canvas)

    def cut(self, reference_point: tuple, angle_degrees: int):
        """
        Cut the shape along a line defined by a reference point and an angle.

        Args:
            reference_point (tuple): tuple of two floats between 0 and 1, indicating the reference point coordinates relative to the bounding box of the image
            angle_degrees (int): angle in degrees for the line to cut the shape

        Returns:
            two CoreShape objects: the two shapes resulting from the cut
        """
        bbox = self.canvas.getbbox()
        ref_point = (
            bbox[0] + reference_point[0] * (bbox[2] - bbox[0]),
            bbox[1] + reference_point[1] * (bbox[3] - bbox[1]),
        )
        angle_radians = np.deg2rad(angle_degrees)

        m = np.tan(angle_radians)
        b = ref_point[1] - m * ref_point[0]

        canvas_array = np.asarray(self.canvas)

        mask = np.all(canvas_array == self.color, axis=2)

        part1_array = np.zeros_like(canvas_array)
        part2_array = np.zeros_like(canvas_array)

        x, y = np.meshgrid(
            np.arange(canvas_array.shape[1]), np.arange(canvas_array.shape[0])
        )
        part1_mask = y < m * x + b
        part1_array[part1_mask & mask] = canvas_array[part1_mask & mask]
        part2_array[~part1_mask & mask] = canvas_array[~part1_mask & mask]

        part1_canvas = Image.fromarray(part1_array, mode="RGBA")
        part2_canvas = Image.fromarray(part2_array, mode="RGBA")

        part1_shape = ShapeCoreFunctions(parent=self.parent, init_img=part1_canvas)
        part2_shape = ShapeCoreFunctions(parent=self.parent, init_img=part2_canvas)

        return part1_shape, part2_shape

    # helper functions ---------------------------------------------------------
    def shrink(self, factor: float):
        """
        shrink the image to the factor specified
        1. the center of the current bounding box is calculated kept constant
        2. the bounding box is cropped and shrunk to the new size
        3. a new image is created with transparent background
        4. the new bounding box is calculated based on the center point
        5. the shrunk image is pasted on the new image
        """
        assert 0 < factor <= 1, "factor must be between 0 and 1"
        # compensate the factor with respect to the size of the image not the sides
        factor = math.sqrt(factor)

        bbox = self.canvas.getbbox()
        cropped = self.canvas.crop(bbox)

        # Calculate the new size after shrinking
        new_size = (int(cropped.width * factor), int(cropped.height * factor))
        shrunk = cropped.resize(new_size, Image.ANTIALIAS)

        new_canvas = Image.new("RGBA", (self.initial_image_width,) * 2, (0, 0, 0, 0))

        # Calculate the center point of the cropped image
        center_x, center_y = (bbox[2] + bbox[0]) // 2, (bbox[3] + bbox[1]) // 2

        # Calculate the new bounding box based on the center point
        new_left = center_x - shrunk.width // 2
        new_upper = center_y - shrunk.height // 2
        new_bbox = (
            new_left,
            new_upper,
            new_left + shrunk.width,
            new_upper + shrunk.height,
        )

        new_canvas.paste(shrunk, new_bbox)
        self.canvas = new_canvas
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
        self.update_self_position()
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
        self.update_self_position()
        return self

    def move_next_to(self, other_shape: "ShapeCoreFunctions", direction_degrees: int):
        "string to angle"

        if type(direction_degrees) == str:
            direction_degrees = {
                "DOWN": 90,
                "UP": 270,
                "LEFT": 180,
                "RIGHT": 0,
                "DOWN_LEFT": 135,
                "DOWN_RIGHT": 45,
                "UP_LEFT": 225,
                "UP_RIGHT": 315,
            }[direction_degrees]

        step = 2
        monitor_other = SizeMonitor()
        offset = 0
        direction_vector = np.array(
            [
                np.cos(np.deg2rad(direction_degrees)),
                np.sin(np.deg2rad(direction_degrees)),
            ]
        )

        bbox_self = self.canvas.getbbox()
        bbox_other = other_shape.canvas.getbbox()

        center_bbox_self = np.array(
            [(bbox_self[0] + bbox_self[2]) / 2, (bbox_self[1] + bbox_self[3]) / 2]
        )
        center_bbox_other = np.array(
            [(bbox_other[0] + bbox_other[2]) / 2, (bbox_other[1] + bbox_other[3]) / 2]
        )
        centering_vector = center_bbox_other - center_bbox_self

        experiment_canvas = new(
            "RGBA", (self.initial_image_width,) * 2, color=(0, 0, 0, 0)
        )
        while not monitor_other.check_stable():
            # paste with transparent background
            experiment_canvas.paste(other_shape.canvas, (0, 0), other_shape.canvas)
            # get a coordinate for self so that the bounding box of self is centered on the bounding box of other
            coordinate_self = centering_vector + (direction_vector * offset)
            experiment_canvas.paste(
                self.canvas, tuple(coordinate_self.astype(int)), self.canvas
            )

            offset += step

            monitor_other.add(get_area_size(experiment_canvas, other_shape.color))
            experiment_canvas = new(
                "RGBA", (self.initial_image_width,) * 2, color=(0, 0, 0, 0)
            )

        offset -= monitor_other.window * step
        coordinate_self = centering_vector + (direction_vector * offset)

        # move the entire canvas to the new position
        canvas_new = new("RGBA", (self.initial_image_width,) * 2, color=(0, 0, 0, 0))
        canvas_new.paste(self.canvas, tuple(coordinate_self.astype(int)), self.canvas)
        self.canvas = canvas_new
        self.draw = Draw(self.canvas)
        self.update_self_position()
        return self

    def move_towards(self, direction_degrees, distance: int):
        if type(direction_degrees) == str:
            direction_degrees = {
                "DOWN": 90,
                "UP": 270,
                "LEFT": 180,
                "RIGHT": 0,
                "DOWN_LEFT": 135,
                "DOWN_RIGHT": 45,
                "UP_LEFT": 225,
                "UP_RIGHT": 315,
            }[direction_degrees]

        direction_vector = np.array(
            [
                np.cos(np.deg2rad(direction_degrees)),
                np.sin(np.deg2rad(direction_degrees)),
            ]
        )
        canvas_new = new("RGBA", (self.initial_image_width,) * 2, color=(0, 0, 0, 0))
        canvas_new.paste(
            self.canvas, tuple((direction_vector * distance).astype(int)), self.canvas
        )
        self.canvas = canvas_new
        self.draw = Draw(self.canvas)
        self.update_self_position()
        return self

    def move_apart_from(self, another_shape, distance: int):
        "move the shape away from another shape by a certain distance"
        bbox_self = self.canvas.getbbox()
        bbox_other = another_shape.canvas.getbbox()

        center_bbox_self = np.array(
            [(bbox_self[0] + bbox_self[2]) / 2, (bbox_self[1] + bbox_self[3]) / 2]
        )
        center_bbox_other = np.array(
            [(bbox_other[0] + bbox_other[2]) / 2, (bbox_other[1] + bbox_other[3]) / 2]
        )
        centering_vector = center_bbox_other - center_bbox_self
        direction_degrees = np.rad2deg(
            np.arctan2(centering_vector[1], centering_vector[0])
        )
        direction_degrees = (direction_degrees + 180) % 360

        self.move_towards(direction_degrees, distance)
        self.update_self_position()
        return self

    def get_distance_from(self, another_shape):
        """get the distance between the center of the bbox of shape and another shape"""
        bbox_self = self.canvas.getbbox()
        bbox_other = another_shape.canvas.getbbox()

        center_bbox_self = np.array(
            [(bbox_self[0] + bbox_self[2]) / 2, (bbox_self[1] + bbox_self[3]) / 2]
        )
        center_bbox_other = np.array(
            [(bbox_other[0] + bbox_other[2]) / 2, (bbox_other[1] + bbox_other[3]) / 2]
        )
        centering_vector = center_bbox_other - center_bbox_self

        return np.linalg.norm(centering_vector)

    # --------------------------------------------------------------------------
    def is_touching(self, another_shape):
        """
        check if the shape overlaps with another shape
        1. convert the canvas to a numpy array
        2. make a boolean mask for the pixels that are not transparent
        3. check if the two boolean masks have any common True values
        4. return the result
        """
        canvas_array_self = np.asarray(self.canvas)
        canvas_array_other = np.asarray(another_shape.canvas)

        mask_self = canvas_array_self[..., 3] > 0
        mask_other = canvas_array_other[..., 3] > 0

        return np.any(mask_self & mask_other)

    def set_color(self, color=None):
        """fill all the non-transparent pixels with the current color"""
        self.color = colors.to_rgba(color, alpha=1) if color else self.color
        self.color = tuple([int(i * 255) for i in self.color])
        canvas_array = np.asarray(self.canvas)

        mask = canvas_array[..., 3] > 0
        fill_array = np.zeros_like(canvas_array)
        fill_array[mask] = self.color
        fill_canvas = Image.fromarray(fill_array, mode="RGBA")
        self.canvas.paste(fill_canvas, (0, 0), fill_canvas)

        return self

    def check_available(self):
        """print the available shapes"""
        available_shapes = [i for i in self.__dir__() if i.startswith("add_")]
        pprint.pprint(available_shapes)

    def register(self):
        self.parent.canvas.paste(self.canvas, (0, 0), self.canvas)
        return self

    def update_self_position(self):
        """
        get the center of the bounding box of the shape and update the position of the shape
        1. get the bounding box of the shape
        2. get the center of the bounding box
        3. normalize the center of the bounding box between 0 and 1 by dividing by the initial image width
        """
        bbox = self.canvas.getbbox()
        center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        self.position = center / self.initial_image_width
        return self
