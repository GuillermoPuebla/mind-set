"""
Base on the Code by Doerig, A., Choung, O. H.:
https://github.com/adriendoerig/Doerig-Bornet-Choung-Herzog-2019/blob/master/Code/Alexnet/batch_maker.py
Used with written permission.
Main changes to the main code:
    - drawPolygon uses the draw.polygon_perimeter which solves some artifacts encountered with the previous method
    - drawCircle now uses the skimage.circle_perimeter which solves some artifacts encountered with the previous method
    - deleted functions we didn't need
    - changed naming convention from CamelCase to snake_case.
    - added optional arguments part for generating dataset from the command line
"""
import shutil
from pathlib import Path
import argparse
import glob
import os
import PIL.Image as Image
import numpy, random
from skimage import draw
from skimage.draw import circle_perimeter
from scipy.ndimage import zoom
from datetime import datetime
from src.utils.misc import apply_antialiasing


def all_test_shapes():
    return (
        seven_shapesgen(5)
        + shapesgen(5)
        + Lynns_patterns()
        + ten_random_patterns()
        + Lynns_patterns()
    )


def shapesgen(max, emptyvect=True):
    if max > 7:
        return

    if emptyvect:
        s = [[]]
    else:
        s = []
    for i in range(1, max + 1):
        s += [[i], [i, i, i], [i, i, i, i, i]]
        for j in range(1, max + 1):
            if j != i:
                s += [[i, j, i, j, i]]

    return s


def seven_shapesgen(max, emptyvect=True):
    if max > 7:
        return

    if emptyvect:
        s = [[]]
    else:
        s = []
    for i in range(1, max + 1):
        s += [[i, i, i, i, i, i, i]]
        for j in range(1, max + 1):
            if j != i:
                s += [[j, i, j, i, j, i, j]]

    return s


def Lynns_patterns():
    squares = [1, 1, 1, 1, 1, 1, 1]
    one_square = [0, 0, 0, 1, 0, 0, 0]
    S = [squares]
    for x in [6, 2]:
        line1 = [x, 1, x, 1, x, 1, x]
        line2 = [1, x, 1, x, 1, x, 1]

        line0 = [x, 1, x, 0, x, 1, x]

        columns = [line1, line1, line1]
        checker = [line2, line1, line2]
        if x == 6:
            special = [1, x, 2, x, 1, x, 1]
        else:
            special = [1, x, 1, x, 6, x, 1]

        checker_special = [line2, line1, special]

        irreg = [[1, x, 1, x, x, 1, 1], line1, [1, 1, x, x, 1, x, 1]]
        cross = [one_square, line1, one_square]
        pompom = [line0, line1, line0]

        S += [line1, columns, checker, irreg, pompom, cross, checker_special]
    return S


def ten_random_patterns(newone=False):
    patterns = numpy.zeros((10, 3, 7), dtype=int)
    if newone:
        basis = [0, 1, 2, 6]
        for pat in range(10):
            for row in range(3):
                for col in range(7):
                    a = numpy.random.choice(basis)
                    patterns[pat][row][col] = a
    else:
        patterns = [
            [[6, 1, 1, 0, 1, 6, 2], [0, 1, 0, 1, 2, 1, 1], [1, 0, 1, 6, 6, 2, 6]],
            [[1, 6, 1, 1, 2, 0, 2], [6, 2, 2, 6, 0, 1, 2], [1, 1, 0, 6, 1, 1, 1]],
            [[1, 6, 1, 2, 2, 0, 2], [1, 0, 6, 1, 2, 2, 6], [2, 2, 0, 1, 0, 2, 1]],
            [[6, 6, 0, 1, 1, 6, 6], [1, 1, 1, 2, 2, 6, 1], [6, 6, 2, 1, 6, 0, 6]],
            [[0, 6, 2, 2, 2, 6, 6], [2, 0, 1, 1, 6, 6, 6], [1, 0, 6, 0, 2, 6, 2]],
            [[2, 1, 1, 6, 2, 6, 2], [6, 1, 0, 6, 1, 2, 1], [1, 6, 0, 2, 1, 2, 6]],
            [[1, 1, 0, 6, 6, 6, 1], [1, 0, 0, 1, 2, 1, 1], [2, 1, 0, 2, 6, 1, 6]],
            [[0, 6, 6, 2, 2, 0, 2], [1, 6, 1, 6, 6, 2, 2], [2, 1, 6, 1, 0, 2, 2]],
            [[6, 1, 2, 6, 1, 0, 1], [0, 1, 6, 2, 0, 6, 2], [1, 0, 1, 2, 6, 6, 6]],
            [[1, 0, 1, 6, 2, 6, 2], [0, 6, 6, 2, 0, 1, 1], [6, 6, 1, 6, 0, 2, 1]],
        ]
        return patterns


class StimMaker:
    def __init__(self, imSize, shapeSize, bar_width):
        self.imSize = imSize
        self.shape_size = shapeSize
        self.bar_width = bar_width
        self.bar_height = int(shapeSize / 4 - bar_width / 4)
        self.offset_height = 1

    def set_shape_size(self, shapeSize):
        self.shape_size = shapeSize
        self.bar_height = int(shapeSize / 4 - barWidth / 4)

    def draw_square(self):
        resize_factor = 1.2
        patch = numpy.zeros((self.shape_size, self.shape_size))

        first_row = int((self.shape_size - self.shape_size / resize_factor) / 2)
        first_col = first_row
        side_size = int(self.shape_size / resize_factor)

        patch[
            first_row : first_row + self.bar_width,
            first_col : first_col + side_size + self.bar_width,
        ] = random.uniform(1 - random_pixels, 1 + random_pixels)
        patch[
            first_row + side_size : first_row + self.bar_width + side_size,
            first_col : first_col + side_size + self.bar_width,
        ] = random.uniform(1 - random_pixels, 1 + random_pixels)
        patch[
            first_row : first_row + side_size + self.bar_width,
            first_col : first_col + self.bar_width,
        ] = random.uniform(1 - random_pixels, 1 + random_pixels)
        patch[
            first_row : first_row + side_size + self.bar_width,
            first_row + side_size : first_row + self.bar_width + side_size,
        ] = random.uniform(1 - random_pixels, 1 + random_pixels)

        return patch

    def draw_circle(self):
        resizeFactor = 1.01
        radius = self.shape_size / (2 * resizeFactor) - 1
        patch = numpy.zeros((self.shape_size, self.shape_size))
        center = (
            int(self.shape_size / 2),
            int(self.shape_size / 2),
        )  # due to discretization, you maybe need add or remove 1 to center coordinates to make it look nice
        rr, cc = circle_perimeter(*center, int(radius))
        patch[rr, cc] = 1
        return patch

    def draw_diamond(self):
        S = self.shape_size
        mid = int(S / 2)
        resizeFactor = 1.00
        patch = numpy.zeros((S, S))
        for i in range(S):
            for j in range(S):
                if i == mid + j or i == mid - j or j == mid + i or j == 3 * mid - i - 1:
                    patch[i, j] = 1

        return patch

    def draw_noise(self):
        S = self.shape_size
        patch = numpy.random.normal(0, 0.1, size=(S // 3, S // 3))
        return patch

    def draw_polygon(self, nSides, phi):
        resizeFactor = 1.0
        patch = numpy.zeros((self.shape_size, self.shape_size))
        center = (self.shape_size // 2, self.shape_size // 2)
        radius = self.shape_size / (2 * resizeFactor) - 1

        rowExtVertices = []
        colExtVertices = []
        rowIntVertices = []
        colIntVertices = []
        for n in range(nSides):
            rowExtVertices.append(
                radius * numpy.sin(2 * numpy.pi * n / nSides + phi) + center[0]
            )
            colExtVertices.append(
                radius * numpy.cos(2 * numpy.pi * n / nSides + phi) + center[1]
            )
            rowIntVertices.append(
                (radius - self.bar_width) * numpy.sin(2 * numpy.pi * n / nSides + phi)
                + center[0]
            )
            colIntVertices.append(
                (radius - self.bar_width) * numpy.cos(2 * numpy.pi * n / nSides + phi)
                + center[1]
            )

        RR, CC = draw.polygon_perimeter(rowExtVertices, colExtVertices)
        patch[RR, CC] = random.uniform(1 - random_pixels, 1 + random_pixels)
        return patch

    def draw_star(self, nTips, ratio, phi):
        resizeFactor = 0.8
        patch = numpy.zeros((self.shape_size, self.shape_size))
        center = (int(self.shape_size / 2), int(self.shape_size / 2))
        radius = self.shape_size / (2 * resizeFactor)

        row_ext_vertices = []
        col_ext_vertices = []
        row_int_vertices = []
        col_int_vertices = []
        for n in range(2 * nTips):
            this_radius = radius
            if not n % 2:
                this_radius = radius / ratio

            row_ext_vertices.append(
                max(
                    min(
                        this_radius * numpy.sin(2 * numpy.pi * n / (2 * nTips) + phi)
                        + center[0],
                        self.shape_size,
                    ),
                    0.0,
                )
            )
            col_ext_vertices.append(
                max(
                    min(
                        this_radius * numpy.cos(2 * numpy.pi * n / (2 * nTips) + phi)
                        + center[1],
                        self.shape_size,
                    ),
                    0.0,
                )
            )
            row_int_vertices.append(
                max(
                    min(
                        (this_radius - self.bar_width)
                        * numpy.sin(2 * numpy.pi * n / (2 * nTips) + phi)
                        + center[0],
                        self.shape_size,
                    ),
                    0.0,
                )
            )
            col_int_vertices.append(
                max(
                    min(
                        (this_radius - self.bar_width)
                        * numpy.cos(2 * numpy.pi * n / (2 * nTips) + phi)
                        + center[1],
                        self.shape_size,
                    ),
                    0.0,
                )
            )

        RR, CC = draw.polygon(row_ext_vertices, col_ext_vertices)
        rr, cc = draw.polygon(row_int_vertices, col_int_vertices)
        patch[RR, CC] = random.uniform(1 - random_pixels, 1 + random_pixels)
        patch[rr, cc] = 0.0

        return patch

    def draw_irreg(self, n_sides_rough, repeat_shape):
        if repeat_shape:
            random.seed(1)

        patch = numpy.zeros((self.shape_size, self.shape_size))
        center = (int(self.shape_size / 2), int(self.shape_size / 2))
        angle = 0  # first vertex is at angle 0

        row_ext_vertices = []
        col_ext_vertices = []
        row_int_vertices = []
        col_int_vertices = []
        while angle < 2 * numpy.pi:
            if (
                numpy.pi / 4 < angle < 3 * numpy.pi / 4
                or 5 * numpy.pi / 4 < angle < 7 * numpy.pi / 4
            ):
                radius = (random.random() + 2.0) / 3.0 * self.shape_size / 2
            else:
                radius = (random.random() + 1.0) / 2.0 * self.shape_size / 2

            row_ext_vertices.append(radius * numpy.sin(angle) + center[0])
            col_ext_vertices.append(radius * numpy.cos(angle) + center[1])
            row_int_vertices.append(
                (radius - self.bar_width) * numpy.sin(angle) + center[0]
            )
            col_int_vertices.append(
                (radius - self.bar_width) * numpy.cos(angle) + center[1]
            )

            angle += (random.random() + 0.5) * (2 * numpy.pi / n_sides_rough)

        RR, CC = draw.polygon(row_ext_vertices, col_ext_vertices)
        rr, cc = draw.polygon(row_int_vertices, col_int_vertices)
        patch[RR, CC] = random.uniform(1 - random_pixels, 1 + random_pixels)
        patch[rr, cc] = 0.0

        if repeat_shape:
            random.seed(datetime.now())

        return patch

    def draw_stuff(self, nLines):
        patch = numpy.zeros((self.shape_size, self.shape_size))

        for n in range(nLines):
            (r1, c1, r2, c2) = numpy.random.randint(self.shape_size, size=4)
            rr, cc = draw.line(r1, c1, r2, c2)
            patch[rr, cc] = random.uniform(1 - random_pixels, 1 + random_pixels)

        return patch

    def draw_vernier(self, offset=None, offset_size=None):
        if offset_size is None:
            offset_size = random.randint(1, int(self.bar_height / 2.0))
        patch = numpy.zeros(
            (2 * self.bar_height + self.offset_height, 2 * self.bar_width + offset_size)
        )
        patch[0 : self.bar_height, 0 : self.bar_width] = 1.0
        patch[
            self.bar_height + self.offset_height :, self.bar_width + offset_size :
        ] = random.uniform(1 - random_pixels, 1 + random_pixels)

        if offset is None:
            if random.randint(0, 1):
                patch = numpy.fliplr(patch)
        elif offset == 1:
            patch = numpy.fliplr(patch)

        full_patch = numpy.zeros((self.shape_size, self.shape_size))
        first_row = int((self.shape_size - patch.shape[0]) / 2)
        first_col = int((self.shape_size - patch.shape[1]) / 2)
        full_patch[
            first_row : first_row + patch.shape[0],
            first_col : first_col + patch.shape[1],
        ] = patch

        return full_patch

    def draw_shape(self, shapeID, offset=None, offset_size=None):
        if shapeID == 0:
            patch = numpy.zeros((self.shape_size, self.shape_size))
        if shapeID == 1:
            patch = self.draw_square()
        if shapeID == 2:
            patch = self.draw_circle()
        if shapeID == 3:
            patch = self.draw_polygon(6, 0)
        if shapeID == 4:
            patch = self.draw_polygon(8, numpy.pi / 8)
        if shapeID == 5:
            patch = self.draw_diamond()
        if shapeID == 6:
            patch = self.draw_star(7, 1.7, -numpy.pi / 14)
        if shapeID == 7:
            patch = self.draw_irreg(15, False)
        if shapeID == 8:
            patch = self.draw_irreg(15, True)
        if shapeID == 9:
            patch = self.draw_stuff(5)
        if shapeID == 10:
            patch = self.draw_noise()

        return patch

    def draw_stim(
        self,
        vernier_ext,
        shape_matrix,
        vernier_in=False,
        offset=None,
        offset_size=None,
        fixed_position=None,
        noise_patch=None,
    ):
        if shape_matrix == None:
            ID = numpy.random.randint(1, 7)
            siz = numpy.random.randint(4) * 2 + 1
            h = numpy.random.randint(2) * 2 + 1
            shape_matrix = numpy.zeros((h, siz)) + ID

        image = numpy.zeros(self.imSize)
        critDist = 0  # int(self.shapeSize/6)
        padDist = int(self.shape_size / 6)
        shape_matrix = numpy.array(shape_matrix)

        if len(shape_matrix.shape) < 2:
            shape_matrix = numpy.expand_dims(shape_matrix, axis=0)

        if shape_matrix.size == 0:  # this means we want only a vernier
            patch = numpy.zeros((self.shape_size, self.shape_size))
        else:
            patch = numpy.zeros(
                (
                    shape_matrix.shape[0] * self.shape_size
                    + (shape_matrix.shape[0] - 1) * critDist
                    + 1,
                    shape_matrix.shape[1] * self.shape_size
                    + (shape_matrix.shape[1] - 1) * critDist
                    + 1,
                )
            )

            for row in range(shape_matrix.shape[0]):
                for col in range(shape_matrix.shape[1]):
                    first_row = row * (self.shape_size + critDist)
                    first_col = col * (self.shape_size + critDist)

                    patch[
                        first_row : first_row + self.shape_size,
                        first_col : first_col + self.shape_size,
                    ] = self.draw_shape(shape_matrix[row, col], offset, offset_size)

        if vernier_in:
            first_row = int(
                (patch.shape[0] - self.shape_size) / 2
            )  # + 1  # small adjustments may be needed depending on precise image size
            first_col = int((patch.shape[1] - self.shape_size) / 2)  # + 1
            patch[
                first_row : (first_row + self.shape_size),
                first_col : first_col + self.shape_size,
            ] += self.draw_vernier(offset, offset_size)
            patch[patch > 1.0] = 1.0

        if fixed_position is None:
            first_row = random.randint(
                padDist, self.imSize[0] - (patch.shape[0] + padDist)
            )
            first_col = random.randint(
                padDist, self.imSize[1] - (patch.shape[1] + padDist)
            )
        else:
            n_elements = [
                max(shape_matrix.shape[0], 1),
                max(shape_matrix.shape[1], 1),
            ]  # because vernier alone has matrix [[]] but 1 element
            first_row = fixed_position[0] - int(
                self.shape_size * (n_elements[0] - 1) / 2
            )  # this is to always have the vernier at the fixed_position
            first_col = fixed_position[1] - int(
                self.shape_size * (n_elements[1] - 1) / 2
            )  # this is to always have the vernier at the fixed_position

        image[
            first_row : first_row + patch.shape[0],
            first_col : first_col + patch.shape[1],
        ] = patch

        min_distance = 0

        if vernier_ext:
            ver_size = self.shape_size
            ver_patch = numpy.zeros((ver_size, ver_size)) + self.draw_vernier(
                offset, offset_size
            )
            x = first_row
            y = first_col

            flag = 0
            while (
                x + ver_size + min_distance >= first_row
                and x <= min_distance + first_row + patch.shape[0]
                and y + ver_size >= first_col
                and y <= first_col + patch.shape[1]
            ):
                x = numpy.random.randint(padDist, self.imSize[0] - (ver_size + padDist))
                y = numpy.random.randint(padDist, self.imSize[1] - (ver_size + padDist))

                flag += 1
                if flag > 15:
                    print("problem in finding space for the extra vernier")

            image[x : x + ver_size, y : y + ver_size] = ver_patch

        if noise_patch is not None:
            image[
                noise_patch[0] : noise_patch[0] + self.shape_size // 2,
                noise_patch[1] : noise_patch[1] + self.shape_size // 2,
            ] = self.draw_noise()

        return image


output_folder = Path("data/gestalt/un_crowding")

[shutil.rmtree(f) for f in glob.glob(str(output_folder / "**")) if os.path.exists(f)]

[
    (output_folder / i / c).mkdir(exist_ok=True, parents=True)
    for i in ["train", "test"]
    for c in ["0", "1"]
]

random_pixels = 0


def generate_all(
    canvas_size,
    num_training_samples,
    num_testing_samples,
    random_size,
    antialiasing,
):
    t = all_test_shapes()

    conditions = ["train", "test"]
    for cond in conditions:
        num_requested_samples = (
            num_training_samples if cond == "train" else num_testing_samples
        )
        samples_per_cond = num_requested_samples // len(t)
        assert (
            samples_per_cond > 0
        ), f"Too many conditions! Increase the number of total {cond} samples to at least {len(t)}"
        print(
            f"You specified {num_requested_samples} for {cond} but to keep the number of sample per subcategory equal, {samples_per_cond * len(t)} samples will be generated ({len(t)} categories, {samples_per_cond} samples per category)"
        ) if samples_per_cond * len(t) != num_requested_samples else None
        vernier_type = [0, 1]
        for v in vernier_type:
            for s in t:
                for n in range(samples_per_cond):
                    shape_size = (
                        random.randint(
                            int(canvas_size[0] * 0.1),
                            canvas_size[0] // 7,
                        )
                        if random_size
                        else canvas_size[0] * 0.08
                    )
                    shape_size -= int(shape_size / 6)

                    rufus = StimMaker(canvas_size, shape_size, bar_width=1)
                    img = rufus.draw_stim(
                        vernier_ext=cond == "train",
                        shape_matrix=s,
                        vernier_in=cond != "train",
                        fixed_position=None,
                        offset=v,
                        offset_size=None,
                        noise_patch=None,
                    )
                    strs = str(s).replace("], ", "nl")
                    code = "".join([i for i in strs if i not in [",", "[", "]", " "]])
                    code = code if code != "" else "none"
                    img = Image.fromarray(img * 255).convert("RGB")
                    img = apply_antialiasing(img) if antialiasing else img
                    img.save(output_folder / cond / str(v) / f"{code}_{n}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--canvas_size",
        "-csize",
        default="224x224",
        help="A string in the format NxM specifying the size of the canvas",
        type=lambda x: tuple([int(i) for i in x.split("x")]),
    )
    parser.add_argument(
        "--num_training_samples",
        "-ntrain",
        default=10000,
        type=int,
    )
    parser.add_argument(
        "--num_testing_samples",
        "-ntest",
        default=5000,
        help="This is the total testing images generated _for each category_",
        type=int,
    )

    parser.add_argument(
        "--random_size",
        "-rnds",
        help="Specify whether the size of the shapes will vary across samples",
        action="store_true",
        default=True,
    )

    parser.add_argument(
        "--no_antialiasing",
        "-nantial",
        dest="antialiasing",
        help="Specify whether we want to disable antialiasing",
        action="store_false",
        default=True,
    )

    args = parser.parse_known_args()[0]
    generate_all(**args.__dict__)
