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
        self.resize_up_resize_down = antialiasing
        self.antialiasing = antialiasing
        self.borders = borders
        self.background = background
        self.canvas_size = canvas_size
        if width is None:
            width = int(np.round(canvas_size[0] * 0.00893))

        if background == "random":
            self.line_col = (0, 0, 0) if line_col is None else line_col

        elif background == (0, 0, 0):
            self.line_col = (255, 255, 255) if line_col is None else line_col

        elif background == (255, 255, 255):
            self.line_col = (0, 0, 0) if line_col is None else line_col

        elif background == "rnd-uniform":
            self.line_col = (255, 255, 255) if line_col is None else line_col

        self.fill = (*self.line_col, 255)
        self.line_args = dict(fill=self.fill, width=width)  # , joint="curve")
        if borders:
            self.borders_width = (
                self.line_args["width"] if borders_width is None else borders_width
            )
        else:
            self.borders_width = 0

    def create_canvas(self, borders=None, size=None, background=None):
        background = self.background if background is None else background
        size = self.canvas_size if size is None else size
        if background == "random":
            img = Image.fromarray(
                np.random.randint(0, 255, (size[1], size[0], 3)).astype(np.uint8),
                mode="RGB",
            )
        elif isinstance(background, tuple):
            img = Image.new("RGB", size, background)  # x and y
        elif background == "rnd-uniform":
            img = Image.new(
                "RGB",
                size,
                (
                    np.random.randint(256),
                    np.random.randint(256),
                    np.random.randint(256),
                ),
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


def get_mask_from_linedrawing(img_path, size):
    linedrawing = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    linedrawing = cv2.resize(
        linedrawing,
        size,
        interpolation=cv2.INTER_AREA,
    )
    mask = np.zeros_like(linedrawing)

    _, binary_linedrawing = cv2.threshold(linedrawing, 240, 255, cv2.THRESH_BINARY_INV)

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
