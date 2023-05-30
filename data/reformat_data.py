"""
Utility: Convert the original released file to our standard jsonl file for further processing
"""
import argparse
import json
import os
from utils.basic_utils import save_jsonl


def normalize_sec(sec):
    return int(sec + 0.5)


def reformat_ego4d_data(split_data, test_split=False):
    """
    Convert the format from JSON files.
    """
    datalist = []
    for video_datum in split_data["videos"]:
        for clip_datum in video_datum["clips"]:
            for ann_datum in clip_datum["annotations"]:
                anno_id = ann_datum['annotation_uid']
                for qid, datum in enumerate(ann_datum["language_queries"]):
                    if "query" not in datum or not datum["query"]:
                        continue
                    temp_dict = {'query': datum["query"],
                                 'query_id': f'{anno_id}_{qid}',
                                 'duration': normalize_sec(clip_datum['video_end_sec'])
                                             - normalize_sec(clip_datum['video_start_sec']),
                                 'clip_id': clip_datum['clip_uid'],
                                 'video_id': video_datum['video_uid'],
                                 'clip_video_start_end': [
                                     normalize_sec(clip_datum['video_start_sec']),
                                     normalize_sec(clip_datum['video_end_sec'])],
                                 }
                    if not test_split:
                        temp_dict["timestamps"] = [datum['clip_start_sec'], datum['clip_end_sec']]
                    datalist.append(temp_dict)
    return datalist


def reformat_mad_data(split_data):
    datalist = []
    for key, value in split_data.items():
        temp_dict = {
            'query': value['sentence'],
            'query_id': key,
            'duration': value["movie_duration"],
            "clip_id": value['movie'],
            "video_id": value['movie'],
            'timestamps': value["timestamps"],
        }
        datalist.append(temp_dict)
    return datalist

def convert_dataset(args):
    """Convert the dataset"""

    for split in ("train", "val", "test" ):
        read_path = args[f"input_{split}_split"]
        print(f"Reading [{split}]: {read_path}")
        with open(read_path, "r") as file_id:
            raw_data = json.load(file_id)
        if args["dset_name"] == "ego4d":
            datalist = reformat_ego4d_data(raw_data, split == "test")
        else:
            datalist = reformat_mad_data(raw_data)
        os.makedirs(args["output_save_path"], exist_ok=True)
        save_path = os.path.join(args["output_save_path"], f"{split}.jsonl")
        print(f"Writing [{split}]: {save_path}")
        save_jsonl(datalist, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_train_split", required=True, help="Path to train split"
    )
    parser.add_argument(
        "--input_val_split", required=True, help="Path to val split"
    )
    parser.add_argument(
        "--input_test_split", required=True, help="Path to test split"
    )
    parser.add_argument(
        "--output_save_path", required=True, help="Path to save the output jsons"
    )
    parser.add_argument("--dset_name", required=True, type=str, choices=["ego4d", 'mad'])
    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))

    convert_dataset(parsed_args)
