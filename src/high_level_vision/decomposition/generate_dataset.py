from src.utils.stimuli.modules.parent import ParentStimuli
from src.utils.stimuli.modules.shapes import Shapes
from src.utils.stimuli.utils.parallel import parallel_args
from itertools import product
import random
from pathlib import Path
import numpy as np
import uuid

moving_distance = 60
shape_size = 0.05
familiar_shapes = ["arc", "circle", "square", "rectangle", "polygon", "triangle"]
unfamiliar_shapes = ["puddle"]

cut_rotation_fun = lambda: random.uniform(0, 360)
rotation_fun = lambda: random.uniform(0, 360)
position_fun = lambda: (random.uniform(0.1, 0.9), random.uniform(0.1, 0.9))

# ------------------------------------------------------
save_path_base = Path("data", "high_level_vision", "decomposition")

# ------------------------------------------------------

combinations_familiar = list(product(familiar_shapes, familiar_shapes))
combinations_familiar = [
    {"shape_1_name": shape_1_name, "shape_2_name": shape_2_name, "path": "familiar"}
    for shape_1_name, shape_2_name in combinations_familiar
]
combinations_familiar *= 10

# ------------------------------------------------------

combinations_unfamiliar = list(product(unfamiliar_shapes, unfamiliar_shapes))
combinations_unfamiliar = [
    {"shape_1_name": shape_1_name, "shape_2_name": shape_2_name, "path": "unfamiliar"}
    for shape_1_name, shape_2_name in combinations_unfamiliar
]
combinations_unfamiliar *= 10

# ------------------------------------------------------


def make_one(params_dict):
    """
    The make one function has to take no arguments and should save the image to a file
    """
    save_path_root = save_path_base / params_dict["path"]

    (save_path_root / "original").mkdir(parents=True, exist_ok=True)
    (save_path_root / "natural").mkdir(parents=True, exist_ok=True)
    (save_path_root / "unnatural").mkdir(parents=True, exist_ok=True)

    shape_1_name, shape_2_name = (
        params_dict["shape_1_name"],
        params_dict["shape_2_name"],
    )
    image_rotation = rotation_fun()
    image_position = position_fun()

    label = f"{shape_1_name}_{shape_2_name}_{image_rotation.__round__(0)}_{image_position[0].__round__(2)}_{image_position[1].__round__(2)}_"
    label += uuid.uuid4().hex[:8].upper()

    #               #----------------#
    # ------------- # original image # ----------------
    #               #----------------#

    # create parent -------------------------------------------
    parent_original = ParentStimuli(target_image_width=224, initial_expansion=4)

    # create shapes -------------------------------------------
    shape_1 = Shapes(parent_original)
    shape_2 = Shapes(parent_original)
    getattr(shape_1, f"add_{shape_1_name}")(**{"size": shape_size})
    getattr(shape_2, f"add_{shape_2_name}")(**{"size": shape_size})
    shape_1.rotate(30)
    shape_2.rotate(30)

    shape_2.move_next_to(shape_1, "LEFT")

    shape_1.register()
    shape_2.register()

    # finalize -------------------------------------------
    parent_original.binary_filter()
    parent_original.move_to(image_position).rotate(image_rotation)
    parent_original.add_background()
    parent_original.shrink()

    # name the file using the size of the shapes and the jastrow factor
    save_path = save_path_root / "original"
    save_path.mkdir(parents=True, exist_ok=True)
    parent_original.canvas.save(save_path / f"{label}.png")

    #               #----------------#
    # ------------- # natural split  # ----------------
    #               #----------------#

    # create parent -------------------------------------------
    parent_natural = ParentStimuli(target_image_width=224, initial_expansion=4)

    # create shapes -------------------------------------------
    shape_1 = Shapes(parent_natural)
    shape_2 = Shapes(parent_natural)
    getattr(shape_1, f"add_{shape_1_name}")(**{"size": shape_size})
    getattr(shape_2, f"add_{shape_2_name}")(**{"size": shape_size})
    shape_1.rotate(30)
    shape_2.rotate(30)

    shape_2.move_next_to(shape_1, "LEFT")
    shape_2.move_apart_from(shape_1, moving_distance)

    shape_1.register()
    shape_2.register()

    # finalize -------------------------------------------
    parent_natural.binary_filter()
    parent_natural.move_to(image_position).rotate(image_rotation)
    parent_natural.add_background()
    parent_natural.shrink()

    # name the file using the size of the shapes and the jastrow factor
    save_path = save_path_root / "natural"
    save_path.mkdir(parents=True, exist_ok=True)
    parent_natural.canvas.save(save_path / f"{label}.png")

    #               #------------------#
    # ------------- # unnatural split  # ----------------
    #               #------------------#

    # create parent -------------------------------------------
    parent_unnatural = ParentStimuli(target_image_width=224, initial_expansion=4)

    # create shapes -------------------------------------------
    shape_1 = Shapes(parent_unnatural)
    shape_2 = Shapes(parent_unnatural)
    getattr(shape_1, f"add_{shape_1_name}")(**{"size": shape_size})
    getattr(shape_2, f"add_{shape_2_name}")(**{"size": shape_size})
    shape_1.rotate(30)
    shape_2.rotate(30)

    shape_2.move_next_to(shape_1, "LEFT")
    piece_1, piece_2 = shape_2.cut(
        reference_point=(0.5, 0.5), angle_degrees=cut_rotation_fun()
    )

    index = np.argmax(
        [piece_1.get_distance_from(shape_1), piece_2.get_distance_from(shape_1)]
    )
    further_piece = [piece_1, piece_2][index]
    closer_piece = [piece_1, piece_2][1 - index]
    further_piece.move_apart_from(closer_piece, moving_distance)

    shape_1.register()
    piece_1.register()
    piece_2.register()

    # finalize -------------------------------------------
    parent_unnatural.binary_filter()
    parent_unnatural.move_to(image_position).rotate(image_rotation)
    parent_unnatural.add_background()
    parent_unnatural.shrink()

    # name the file using the size of the shapes and the jastrow factor
    save_path = save_path_root / "unnatural"
    save_path.mkdir(parents=True, exist_ok=True)
    parent_unnatural.canvas.save(save_path / f"{label}.png")


if __name__ == "__main__":
    parallel_args(make_one, combinations_familiar, progress_bar=True)
    parallel_args(make_one, combinations_unfamiliar, progress_bar=True)
