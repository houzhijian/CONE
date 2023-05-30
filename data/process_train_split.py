"""
Utility: Remove some train samples, e.g., start timestamp is equal to end timestamp
"""

import math
import os.path
from utils.basic_utils import load_jsonl, save_jsonl
import argparse


def pre_filtering_mad(train_split):
    data = load_jsonl(train_split)
    new_data = []
    for item in data:
        video_duration = item["duration"]
        start, end = item["timestamps"]
        # start timestamp is smaller than zero
        if start < 0:
            print(0, item)
            continue
        # start timestamp is bigger than video duration
        if start >= video_duration:
            print(1, item)
            continue
        if start == end:
            print(2, item)
            continue
        new_data.append(item)

    print("full train split len: ", len(data))
    print("after processing train split len: ", len(new_data))

    filename = os.path.splitext(os.path.basename(train_split))[0]
    dir = os.path.dirname(train_split)
    save_jsonl(new_data, os.path.join(dir, filename + "_v1.jsonl"))


def pre_filtering_ego4d(train_split):
    data = load_jsonl(train_split)
    new_data = []
    for item in data:
        video_duration = item["clip_video_start_end"][1] - item["clip_video_start_end"][0]
        start, end = item["timestamps"]
        # start timestamp is bigger than video duration
        if start >= video_duration or start >= video_duration * 479.895 / 480:
            print(1, item)
            continue
        # start timestamp is equal to end timestamp
        if start == end:
            print(2, item)
            continue
        # no negative window
        if start < 120 and end > video_duration - 60:
            print(3, item)
            continue
        new_data.append(item)

    print("full train split len: ", len(data))
    print("after processing train split len: ", len(new_data))

    filename = os.path.splitext(os.path.basename(train_split))[0]
    dir = os.path.dirname(train_split)
    save_jsonl(new_data, os.path.join(dir, filename + "_v1.jsonl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dset_name", required=True, type=str, choices=["ego4d", 'mad'])
    parser.add_argument(
        "--train_split", required=True, help="Path to train split"
    )
    args = parser.parse_args()
    if args.dset_name == "ego4d":
        pre_filtering_ego4d(args.train_split)
    else:
        pre_filtering_mad(args.train_split)

