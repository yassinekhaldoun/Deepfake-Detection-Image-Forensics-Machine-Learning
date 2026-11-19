from PIL import Image
import numpy as np

def load_image(path):
    with Image.open(path) as image:
        image=np.asarray(image.convert('RGB'), dtype=np.float32)
        return image

def center_crop(image, crop_size=512):
    h, w = image.shape[:2]

    if h < crop_size or w < crop_size:
        raise ValueError("Image is smaller than the crop size")

    top = (h - crop_size) // 2
    left = (w - crop_size) // 2

    return image[
        top:top + crop_size,
        left:left + crop_size
    ]

