import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage import filters
import tyro
import warnings

warnings.filterwarnings("ignore")

import ansi


@dataclass
class Args:
    image_path: str
    """The path to the image"""
    image_format: str = "png"
    """The format of the image"""
    color: bool = True
    """Print in color"""
    width: int = 30
    """The width of the printed ascii image"""
    simplified: bool = True
    """The resolution of the ascii character map"""


luminosity_table_simplified = " .:-=+*#%@"
luminosity_table = r" `.-':_,^=;><+!rc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@"


def read_image(fpath, format):
    im = plt.imread(fpath, format=format)
    # stretch the image horizontally because the stdout puts some buffer above and below
    im = resize(im, (im.shape[0], im.shape[1] * 2), anti_aliasing=True)
    return im


def resize_image(image, width):
    w, _ = image.shape[:2]
    scale_factor = width / w
    low_res_im = resize(
        image,
        (image.shape[0] * scale_factor, image.shape[1] * scale_factor),
        anti_aliasing=True,
    )
    return low_res_im


def get_ascii_edge_from_angle(angle):
    if angle % 180 < 22.5:
        return "-"
    elif angle % 180 < 67.5:
        return "/"
    elif angle % 180 < 112.5:
        return "|"
    elif angle % 180 < 157.5:
        return "\\"
    else:
        return "-"


def main():
    args = tyro.cli(Args)
    edge_min = 0.1
    im = read_image(args.image_path, args.image_format)
    im = rgb2gray(im)
    sobel_x = filters.sobel_h(im)
    sobel_y = filters.sobel_v(im)
    im = resize_image(im, args.width)
    sobel_x = resize_image(sobel_x, args.width)
    sobel_y = resize_image(sobel_y, args.width)
    if args.simplified:
        luma_table = luminosity_table_simplified
    else:
        luma_table = luminosity_table
    # filtered = bw_im  # filters.difference_of_gaussians(bw_im, low_sigma=1, high_sigma=2)

    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    edge_direction = np.degrees(np.arctan2(sobel_y, sobel_x))

    rows, cols = im.shape[:2]
    for i in range(rows):
        for j in range(cols):
            n = len(luma_table)
            idx = int((n - 1) * im[i, j])
            char = luma_table[idx]
            if edge_magnitude[i, j] > edge_min:
                char = get_ascii_edge_from_angle(edge_direction[i, j])
            print(char, end="")
        print()


if __name__ == "__main__":
    main()
