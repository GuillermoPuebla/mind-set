import uuid
import csv
from tqdm import tqdm
from pathlib import Path
from src.color_picker.utils import ShapeConfigs, ColorPickerStimuli, add_arrow

DEFAULTS = {
    "num_samples": 50000,
    "output_folder": Path("data", "color_picker_train"),
    "canvas_size": (224, 224),
}


def generate_all(
    num_samples=DEFAULTS["num_samples"],
    output_folder=DEFAULTS["output_folder"],
    canvas_size=DEFAULTS["canvas_size"],
):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    shape_configs = ShapeConfigs()

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "Pixel Color"])

        for i in tqdm(range(num_samples)):
            img = ColorPickerStimuli(canvas_size)

            for step in range(20):  # abandon the checks, just dump the shapes!
                shape_configs._refresh()
                random_shape_config = shape_configs._return_shape_config()
                getattr(img, "add_" + random_shape_config["shape"])(
                    **random_shape_config["parameters"]
                )

            propose_coordinates = img._propose_arrow_coord()

            while img._count_colors_withing_circle(propose_coordinates) > 1:
                propose_coordinates = img._propose_arrow_coord()

            pixel_color = img._get_pixel_color(propose_coordinates)
            # pixel_color = img._get_pixel_color((0.5, 0.5))  # for center color
            img.canvas = add_arrow(img.canvas, propose_coordinates)
            image_name = str(uuid.uuid4().hex) + ".png"
            img._shrink_and_save(save_as=output_folder / image_name)
            writer.writerow([(output_folder / image_name).__str__(), pixel_color])


if __name__ == "__main__":
    generate_all(100, Path("test"), (224, 224))
