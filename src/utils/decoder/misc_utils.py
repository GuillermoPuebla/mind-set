from torchvision.transforms import functional as F
import random


class AffineTransform:
    def __init__(self, translate_b, rotate_b, scale_b) -> None:
        self.translate_b = translate_b
        self.rotate_b = rotate_b
        self.scale_b = scale_b

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
        )
