import os.path
import tqdm
import lmdb
import msgpack
import msgpack_numpy
import io
import numpy as np
import torch

def dumps_msgpack(dump):
    return msgpack.dumps(dump, use_bin_type=True)


def dumps_npz(dump, compress=False):
    with io.BytesIO() as writer:
        if compress:
            np.savez_compressed(writer, **dump, allow_pickle=True)
        else:
            np.savez(writer, **dump, allow_pickle=True)
        return writer.getvalue()


root_dir_list = [
    "/s1_md0/leiji/v-zhijian/ego4d_data/nlq_egoclip_video_1.875fps",
    ]


feature_save_path = "/s1_md0/leiji/v-zhijian/ego4d_nlq_data_for_cone/offline_lmdb/egovlp_video_feature_1.875fps"
output_clip_env = lmdb.open(feature_save_path, map_size=1099511627776)


for root in root_dir_list:
    for filename in tqdm.tqdm(os.listdir(root)):
        key = os.path.splitext(filename)[0]
        value = torch.load(os.path.join(root,filename))
        with output_clip_env.begin(write=True) as output_txn:
            features_dict = {"features": np.array(value).astype(np.float32)}
            feature_dump = dumps_npz(features_dict, compress=True)
            output_txn.put(key=key.encode(), value=feature_dump)
