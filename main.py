from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
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
luminosity_scale_simplified = np.linspace(0, 1, len(luminosity_table_simplified))
luminosity_table = r" `.-':_,^=;><+!rc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@"
luminosity_scale = [0, 0.0751, 0.0829, 0.0848, 0.1227, 0.1403, 0.1559, 0.185, 0.2183, 0.2417, 0.2571, 0.2852, 0.2902,
                    0.2919, 0.3099, 0.3192, 0.3232, 0.3294, 0.3384, 0.3609, 0.3619, 0.3667, 0.3737, 0.3747, 0.3838,
                    0.3921, 0.396, 0.3984, 0.3993, 0.4075, 0.4091, 0.4101, 0.42, 0.423, 0.4247, 0.4274, 0.4293, 0.4328,
                    0.4382, 0.4385, 0.442, 0.4473, 0.4477, 0.4503, 0.4562, 0.458, 0.461, 0.4638, 0.4667, 0.4686, 0.4693,
                    0.4703, 0.4833, 0.4881, 0.4944, 0.4953, 0.4992, 0.5509, 0.5567, 0.5569, 0.5591, 0.5602, 0.5602, 0.565,
                    0.5776, 0.5777, 0.5818, 0.587, 0.5972, 0.5999, 0.6043, 0.6049, 0.6093, 0.6099, 0.6465, 0.6561, 0.6595,
                    0.6631, 0.6714, 0.6759, 0.6809, 0.6816, 0.6925, 0.7039, 0.7086, 0.7235, 0.7302, 0.7332, 0.7602, 0.7834,
                    0.8037, 0.9999]

def read_image(fpath, format):
    im = plt.imread(fpath, format=format)
    # stretch the image horizontally because the stdout puts some buffer above and below
    im = resize(im, (im.shape[0], im.shape[1]*2), anti_aliasing=True)
    return im

def resize_image(image, width):
    w, _ = image.shape[:2]
    scale_factor = width / w
    low_res_im = resize(image, (image.shape[0] * scale_factor, image.shape[1] * scale_factor), anti_aliasing=True)
    return low_res_im

def find_index(x, luminosity_scale):
    l = 0
    r = len(luminosity_scale) - 1
    while l <= r:
        m = (l + r) // 2
        m_val = luminosity_scale[m]
        if x < m_val:
            r = m - 1
        elif x > m_val:
            l = m + 1
        else:
            return l
    return l

def main():
    args = tyro.cli(Args)
    im = read_image(args.image_path, args.image_format)
    im = resize_image(im, args.width)

    for i in range(im.shape[0]-1):
        for j in range(im.shape[1]-1):
            val = im[i, j, :3].mean()
            r, g, b = list(map(int, 255 * im[i, j, :3]))
            alpha = im[i, j, 3]
            if alpha < 0.9:
                ansi.rich_print(" ", end="")
            else:
                if args.color:
                    ansi.rich_print(f"[{r};{g};{b}]@", end="")
                else:
                    if args.simplified:
                        idx = find_index(val, luminosity_scale_simplified)
                        char = luminosity_table_simplified[idx]
                    else:
                        idx = find_index(val, luminosity_scale)
                        char = luminosity_table[idx]
                    ansi.rich_print(f"{char}", end="")
        print()

if __name__ == "__main__":
    main()

