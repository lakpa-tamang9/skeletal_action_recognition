import argparse
import os

import natsort
from moviepy.editor import *


def reduce_video_resolution(path):
    output_root = ("/").join(path.split("/")[:-1])
    output_path = os.path.join(output_root, "reduced_resolution")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    files = os.listdir(path)
    for file in natsort.natsorted(files):
        filename = file.split(".")[0]

        clip = VideoFileClip(os.path.join(path, file))

        final = clip.fx(vfx.resize, width=1080, height=720)

        final.write_videofile(f"{output_path}/{filename}.mp4")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="Path to the video file whose \
        video resolution needs to be reduced",
    )
    args = parser.parse_args()
    path = args.video_dir
    reduce_video_resolution(path)


if __name__ == "__main__":
    main()
