"""
Utility: extract textual token feature for Ego4d-NLQ
"""
import torch
from utils.basic_utils import load_jsonl
import numpy as np
import tqdm
import msgpack
import io
from clip_extractor import ClipFeatureExtractor
from torch.utils.data import DataLoader, Dataset
import argparse
import os
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def dumps_msgpack(dump):
    return msgpack.dumps(dump, use_bin_type=True)


def dumps_npz(dump, compress=False):
    with io.BytesIO() as writer:
        if compress:
            np.savez_compressed(writer, **dump, allow_pickle=True)
        else:
            np.savez(writer, **dump, allow_pickle=True)
        return writer.getvalue()


class SingleSentenceDataset(Dataset):
    def __init__(self, input_datalist, block_size=512, debug=False):
        self.max_length = block_size
        self.debug = debug
        self.examples = input_datalist

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        self.examples[index]['query'] = self.process_question(self.examples[index]['query'])
        return self.examples[index]

    def process_question(self,question):
        """Process the question to make it canonical."""
        return question.strip(".").strip(" ").strip("?").lower() + "?"


def pad_collate(data):
    batch = {}
    for k in data[0].keys():
        batch[k] = [d[k] for d in data]
    return batch


def extract_ego4d_text_feature(args):
    format = "/s1_md0/leiji/v-zhijian/CONE/data/ego4d_data/%s.jsonl"
    split_list = ['test']  # ['train', 'test', 'val']
    total_data = []
    for split in split_list:
        filename = format % split
        data = load_jsonl(filename)
        total_data.extend(data)
    print(len(total_data))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Build models...")
    clip_model_name_or_path = "ViT-B/32"
    feature_extractor = ClipFeatureExtractor(
        framerate=30, size=224, centercrop=True,
        model_name_or_path=clip_model_name_or_path, device=device
    )

    dataset = SingleSentenceDataset(input_datalist=total_data)

    eval_dataloader = DataLoader(dataset, batch_size=60, collate_fn=pad_collate)

    feature_dict = {}

    for batch in tqdm.tqdm(eval_dataloader, desc="Evaluating", total=len(eval_dataloader)):
        query_id_list = batch["query_id"]
        query_list = batch["query"]
        token_features, text_eot_features = feature_extractor.encode_text(query_list)

        for i in range(len(query_list)):
            token_feat = np.array(token_features[i].detach().cpu()).astype(np.float32)
            # if i == 0:
            #     print("query: ", query_list[i])
            #     print("query tokenize 1: ", _tokenizer.bpe(query_list[i]))
            #     encode_text = _tokenizer.encode(query_list[i])
            #     print("query tokenize 2: ", encode_text)
            #     print("query tokenize idx: ", clip.tokenize(query_list[i]))
            #     print("decoder query: ", _tokenizer.decode(encode_text))
            #     print("token_feat: ", token_feat.shape)

            feature_dict[query_id_list[i]] = token_feat

    for key, value in tqdm.tqdm(feature_dict.items()):
        np.save(os.path.join(args.feature_output_path, key), value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature_output_path", required=True, help="Path to train split"
    )  # "/s1_md0/leiji/v-zhijian/MAD_data/CLIP_text_features"
    args = parser.parse_args()
    extract_ego4d_text_feature(args)
