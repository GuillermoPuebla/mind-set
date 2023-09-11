from torchvision.transforms import functional as F
import random


class AffineTransform:
    def __init__(self, translate_b, rotate_b, scale_b, fill_color=None) -> None:
        self.translate_b = translate_b
        self.rotate_b = rotate_b
        self.scale_b = scale_b
        self.fill_color = [0, 0, 0] if fill_color is None else fill_color

    def __call__(self, img):
        return F.affine(
            img,
            angle=random.uniform(*self.rotate_b),
            translate=(
                random.uniform(*self.translate_b),
                random.uniform(*self.translate_b),
            ),
            scale=random.uniform(*self.scale_b),
            shear=0,
            fill=self.fill_color,
        )
