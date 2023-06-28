import PIL
import cv2
import numpy as np
from PIL import Image, ImageDraw


class DrawShape:
    def __init__(
        self,
        background="black",
        canvas_size: tuple = (224, 224),
        width: int = None,
        borders: bool = False,
        line_col: tuple = None,
        antialiasing: bool = False,
        borders_width: int = None,
    ):
        self.line_col = None
        self.resize_up_resize_down = antialiasing
        self.antialiasing = antialiasing
        self.borders = borders
        self.background_mode = (
            background if isinstance(background, str) else "one_color"
        )
        self.background = background if isinstance(background, tuple) else None
        self.canvas_size = canvas_size
        if width is None:
            width = int(np.round(canvas_size[0] * 0.00893))

        if self.background_mode == "random":
            self.line_col = (0, 0, 0) if line_col is None else line_col
        elif self.background_mode == "rnd-uniform":
            self.line_col = (255, 255, 255) if line_col is None else line_col
        elif self.background_mode == "one_color":
            if self.background == (0, 0, 0):
                self.line_col = (255, 255, 255) if line_col is None else line_col

            elif self.background == (255, 255, 255):
                self.line_col = (0, 0, 0) if line_col is None else line_col

        self.line_col = (0, 0, 0) if self.line_col is None else self.line_col
        self.fill = (*self.line_col, 255)
        self.line_args = dict(fill=self.fill, width=width)  # , joint="curve")
        if borders:
            self.borders_width = (
                self.line_args["width"] if borders_width is None else borders_width
            )
        else:
            self.borders_width = 0

    def create_canvas(self, borders=None, size=None, background=None):
        if background is not None:
            background_mode = background if isinstance(background, str) else "one_color"
            background = background if isinstance(background, tuple) else None
        else:
            background_mode = self.background_mode
            background = self.background
        size = self.canvas_size if size is None else size
        if background_mode == "one_color":
            if isinstance(background, tuple):
                img = Image.new("RGB", size, background)  # x and y

        if background_mode == "random":
            img = Image.fromarray(
                np.random.randint(0, 255, (size[1], size[0], 3)).astype(np.uint8),
                mode="RGB",
            )

        elif background_mode == "rnd-uniform":
            self.background = (
                np.random.randint(256),
                np.random.randint(256),
                np.random.randint(256),
            )
            img = Image.new(
                "RGB",
                size,
                self.background,
            )

        borders = self.borders if borders is None else borders
        if borders:
            draw = ImageDraw.Draw(img)
            draw.line(
                [(0, 0), (0, img.size[0])],
                fill=self.line_args["fill"],
                width=self.borders_width,
            )
            draw.line(
                [(0, 0), (img.size[0], 0)],
                fill=self.line_args["fill"],
                width=self.borders_width,
            )
            draw.line(
                [(img.size[0] - 1, 0), (img.size[0] - 1, img.size[1] - 1)],
                fill=self.line_args["fill"],
                width=self.borders_width,
            )
            draw.line(
                [(0, img.size[0] - 1), (img.size[0] - 1, img.size[1] - 1)],
                fill=self.line_args["fill"],
                width=self.borders_width,
            )

        return img

    def circle(self, draw, center, radius: int, fill=None):
        if not fill:
            fill = self.fill
        draw.ellipse(
            (
                center[0] - radius + 1,
                center[1] - radius + 1,
                center[0] + radius - 1,
                center[1] + radius - 1,
            ),
            fill=fill,
            outline=None,
        )

    @classmethod
    def resize_up_down(cls, fun):
        """
        :return: Decorate EVERY FUNCTION that returns an image with @DrawShape.resize_up_down if you want to enable antialiasing (which you should!)
        """

        def wrap(self, *args, **kwargs):
            if self.antialiasing:
                ori_self_width = self.line_args["width"]
                self.line_args["width"] = self.line_args["width"] * 2
                original_img_size = self.canvas_size
                self.canvas_size = tuple(np.array(self.canvas_size) * 2)

            im = fun(self, *args, **kwargs)

            if self.antialiasing:
                im = im.resize(original_img_size)
                self.line_args["width"] = ori_self_width
                self.canvas_size = original_img_size
            return im

        return wrap


def get_mask_from_linedrawing(opencv_img):
    """
    This function assumed the background of the linedrawing is white, and the stroke is black/grayscale.
    :param opencv_img:
    :return:
    """
    mask = np.zeros_like(opencv_img)

    _, binary_linedrawing = cv2.threshold(opencv_img, 240, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(
        binary_linedrawing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    mask = Image.fromarray(mask).convert("L")

    linedrawing_pil = Image.fromarray(binary_linedrawing)

    mask = Image.merge(
        "RGBA", (linedrawing_pil, linedrawing_pil, linedrawing_pil, mask)
    )
    return mask


def resize_image_keep_aspect_ratio(opencv_img, max_side_length):
    # Calculate new dimensions while keeping the aspect ratio
    height, width = opencv_img.shape
    aspect_ratio = float(width) / float(height)

    if height > width:
        new_height = max_side_length
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = max_side_length
        new_height = int(new_width / aspect_ratio)

    resized_img = cv2.resize(
        opencv_img, (new_width, new_height), interpolation=cv2.INTER_AREA
    )
    # background = np.bincount(resized_img.flatten()).argmax()
    # canvas = np.ones(canvas_size) * background
    # canvas[: resized_img.shape[0], : resized_img.shape[1]] = resized_img
    return resized_img


def paste_linedrawing_onto_canvas(
    opencv_linedrawing, canvas, stroke_color
) -> PIL.Image:
    """
    This function assumes the linedrawing is an opencv_img with black stroke on a white background. Returns a PIL.Image
    """

    mask = get_mask_from_linedrawing(opencv_linedrawing)

    canvas_foreg_text = Image.new(
        "RGB", (opencv_linedrawing.shape[1], opencv_linedrawing.shape[0]), stroke_color
    )

    canvas.paste(
        canvas_foreg_text,
        (
            canvas.size[0] // 2 - mask.size[0] // 2,
            canvas.size[1] // 2 - mask.size[1] // 2,
        ),
        mask=mask,
    )
    return canvas
