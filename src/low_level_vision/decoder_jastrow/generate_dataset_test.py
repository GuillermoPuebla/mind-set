from src.utils.stimuli.modules.parent import ParentStimuli
from src.utils.stimuli.modules.shapes import Shapes
from src.utils.stimuli.utils.parallel import parallel_args
import uuid
import numpy as np
from pathlib import Path
from itertools import product


# define functions to generate random parameters
arc_fun = lambda: np.random.uniform(45, 60, 10)
width_fun = lambda: np.random.uniform(3e-2, 2e-1, 10)
size_fun = lambda: np.random.uniform(1e-2, 6e-2, 10)

combination = list(product(arc_fun(), width_fun(), size_fun()))
combination = [
    {"arc": arc, "width": width, "size": size} for arc, width, size in combination
]


def make_one(params_dict):
    """
    The make one function has to take no arguments and should save the image to a file
    """
    for on_top in ["red", "blue"]:
        for smaller in [0, 0.05, 0.1, 0.15, 0.2, 0.25]:  # percentage of shrinkage
            arc, width, size = (
                params_dict["arc"],
                params_dict["width"],
                params_dict["size"],
            )

            # create parent -------------------------------------------
            parent = ParentStimuli(target_image_width=224)

            # create shapes -------------------------------------------
            arc_1 = Shapes(parent=parent)
            arc_1.add_arc(size=size, arc=arc, width=width)

            arc_2 = Shapes(parent=parent)
            arc_2.add_arc(size=size, arc=arc, width=width)

            arc_1.shrink(1 - smaller)

            arc_1.move_next_to(arc_2, "UP")

            arc_1.set_color(on_top).register()
            arc_2.set_color({"red": "blue", "blue": "red"}[on_top]).register()

            # finalise ------------------------------------------------
            parent.center_shapes()
            parent.add_background()
            parent.shrink()

            try:
                assert parent.count_pixels("red") > 0, "red pixels not found"
                assert parent.count_pixels("blue") > 0, "blue pixels not found"
            except AssertionError:
                return

            label = f"{arc.round(3)}_{width.round(3)}_{size.round(3)}"

            save_path = Path(
                "data",
                "low_level_vision",
                "jastrow_test",
                f"{on_top}_on_top_{smaller}_smaller",
            )

            save_path.mkdir(parents=True, exist_ok=True)
            parent.canvas.save(save_path / f"{label}_{uuid.uuid4().hex[:8]}.png")


if __name__ == "__main__":
    parallel_args(make_one, combination, progress_bar=True)
