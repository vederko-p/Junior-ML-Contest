
import numpy as np
from PIL import Image


def open_image(img_path: str) -> np.array:
    """Open image and transform to [0, 1] values."""
    img = Image.open(img_path)
    array_img = np.expand_dims(np.asarray(img), 2)
    return array_img / 255


def custom_crop(image: np.array) -> np.array:
    """Crop image."""
    if (image.shape[0] == 256) and (image.shape[1] == 256):
        return image[192-42:192+24, 128-80:128+80]
    elif (image.shape[0] == 128) and (image.shape[1] == 128):
        return image[96-21:96+12, 64-40:64+40]
    else:
        raise Exception('Wrong size of image. Should be 128x128 or 256x256.')
