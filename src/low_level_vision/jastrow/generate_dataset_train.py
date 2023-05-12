"""
The output images are already filtered that images that may trigger the Jastrow illusion
are removed. The label is the size differences between the shapes (red - blue) if you
di not run the normalizer; if you did, the label will be between -1 and 1 where larger
number indicates relatively larger red shape.
"""


from src.utils.stimuli.modules.parent import ParentStimuli
from src.utils.stimuli.modules.shapes import Shapes
import random
import uuid
import math
import numpy as np
from pathlib import Path
from src.utils.stimuli.utils.parallel import parallel
from src.utils.stimuli.utils.normalize import normalize_folder


class JastrowParent(ParentStimuli):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _compute_jastrow_factor(self):
        """
        Since we need a way to approximate how much an image will trigger the Jastrow illusion without having to
        actually show the image to a human, I propose the following method - we use a "Jastrow" factor, which is a
        number between 0 and 1, where larger numbers indicate a higher likelihood of triggering the illusion.

        The Jastrow factor is calculated as the product of the rotation similarity and the spatial similarity.
        """
        assert (
            len(self.contained_shapes) == 2
        ), "Make sure there are only two arc shapes"
        assert all(
            [shape.shape == "arc" for shape in self.contained_shapes]
        ), "Shapes must be arc"

        rotations = [shape.rotation for shape in self.contained_shapes]
        positions = [shape.position for shape in self.contained_shapes]

        # calculate the rotation similarity using the cosine similarity
        rotation_similarity = self.get_rotation_similarity(*rotations)

        # calculate the spatial similarity by calculating the distance between the two shapes and dividing by the max distance
        self.position_similarity = np.linalg.norm(
            np.array(positions[0]) - np.array(positions[1])
        ) / math.sqrt(2)
        self.position_similarity = 1 - self.position_similarity

        # calculate the Jastrow factor as the product of rotation similarity and spatial similarity
        jastrow_factor = rotation_similarity * self.position_similarity
        return jastrow_factor

    def get_rotation_similarity(self, rotation1, rotation2):
        cos_sim = math.cos(math.radians(rotation1 - rotation2))
        return (cos_sim + 1) / 2


# define functions to generate random parameters
position_fun = lambda: (random.uniform(0.1, 0.9), random.uniform(0.1, 0.9))
rotation_fun = lambda: random.uniform(0, 360)
arc_fun = lambda: random.uniform(45, 60)
width_fun = lambda: random.uniform(3e-2, 2e-1)
size_fun = lambda: random.uniform(1e-2, 6e-2)


def make_one(*args):
    """
    The make one function has to take no arguments and should save the image to a file
    """

    # create parent -------------------------------------------
    parent = JastrowParent(target_image_width=224, initial_expansion=4)

    # create shapes -------------------------------------------
    arc_1 = Shapes(parent=parent)
    arc_1.add_arc(size=size_fun(), arc=arc_fun(), width=width_fun())
    arc_1.move_to(position_fun()).rotate(rotation_fun())

    arc_2 = Shapes(parent=parent)
    arc_2.add_arc(size=size_fun(), arc=arc_fun(), width=width_fun())
    arc_2.move_to(position_fun()).rotate(rotation_fun())

    # paste make sure the shapes don't overlap ----------------
    while arc_1.is_touching(arc_2):
        arc_1.move_to(position_fun()).rotate(rotation_fun())
        arc_2.move_to(position_fun()).rotate(rotation_fun())

    arc_1.set_color("red").register()
    arc_2.set_color("blue").register()

    # finalise ------------------------------------------------
    jastrow_factor = parent._compute_jastrow_factor().round(3)

    # filter out images with a jastrow factor greater than 0.7
    # This is an arbitrary threshold, but it seems to work well
    if jastrow_factor > 0.7:
        return

    parent._add_background()
    parent._shrink()

    # label the image based on the size of the shapes
    label = parent.count_pixels(color="red") - parent.count_pixels(color="blue")

    # name the file using the size of the shapes and the jastrow factor
    save_path = Path("data", "low_level_vision", "jastrow_train")
    # save_path = Path("E:\\") / "mindset_data" / "jastrow_train"
    save_path.mkdir(parents=True, exist_ok=True)
    parent.canvas.save(save_path / f"{label}_{uuid.uuid4().hex[:8]}.png")


if __name__ == "__main__":
    parallel(make_one, 1000, progress_bar=True)
    # normalize_folder(Path("E:\\") / "mindset_data" / "jastrow_train")
