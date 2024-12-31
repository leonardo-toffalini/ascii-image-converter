import cv2
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage import filters
import tyro
import warnings
import time

warnings.filterwarnings("ignore")


@dataclass
class Args:
    video_path: str
    """The path to the video"""
    width: int = 60
    """Target width"""
    fps: int = 30
    """Target frames per second"""


luminosity_table_simplified = " .:-=+*#%@"
luminosity_table = r" `.-':_,^=;><+!rc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@"


def read_image(fpath, format):
    im = plt.imread(fpath, format=format)
    # stretch the image horizontally because the stdout puts some buffer above and below
    im = resize(im, (im.shape[0], 5 * im.shape[1]), anti_aliasing=True)
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


def get_char(x, luma_table):
    n = len(luma_table)
    idx = int((n - 1) * x)
    return luma_table[idx]


get_char_vectorized = np.vectorize(get_char)
get_ascii_edge_from_angle_vectorized = np.vectorize(get_ascii_edge_from_angle)


def convert_and_print_ascii(im, args, edge_min):
    im = rgb2gray(im)
    filtered = filters.difference_of_gaussians(im, low_sigma=1, high_sigma=2)
    sobel_x = filters.sobel_h(im)
    sobel_y = filters.sobel_v(im)
    im = resize_image(im, args.width)
    sobel_x = resize_image(sobel_x, args.width)
    sobel_y = resize_image(sobel_y, args.width)
    luma_table = luminosity_table_simplified

    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    edge_direction = np.degrees(np.arctan2(sobel_y, sobel_x))

    char_arr = get_char_vectorized(im, luma_table)
    edge_arr = get_ascii_edge_from_angle_vectorized(edge_direction)
    temp_arr = np.where(edge_magnitude > edge_min, edge_arr, np.nan)
    char_arr = np.where(temp_arr == "nan", char_arr, temp_arr)

    for row in char_arr:
        print("".join(row))


def main():
    args = tyro.cli(Args)
    edge_min = 0.1
    FPS = args.fps
    FRAMETIME = 1 / FPS

    # Path to the .mp4 video
    video_path = args.video_path
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video at {args.video_path}.")
    else:
        while True:
            frame_start = time.time()
            ret, frame = cap.read()
            if not ret:
                print("End of video.")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = resize(
                frame, (frame.shape[0], 2 * frame.shape[1]), anti_aliasing=False
            )
            convert_and_print_ascii(frame, args, edge_min)
            frame_end = time.time()

            # wait if the rendering took less than 1/FPS seconds
            render_time = frame_end - frame_start
            diff = FRAMETIME - render_time
            if diff > 0:
                time.sleep(diff)

    cap.release()


if __name__ == "__main__":
    main()
