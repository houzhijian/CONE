"""
Utility: Convert official MAD H5 visual file to the lmdb file
"""


import os.path
import h5py
import tqdm
import lmdb
import msgpack
import msgpack_numpy
import io
import numpy as np


def dumps_msgpack(dump):
    return msgpack.dumps(dump, use_bin_type=True)


def dumps_npz(dump, compress=False):
    with io.BytesIO() as writer:
        if compress:
            np.savez_compressed(writer, **dump, allow_pickle=True)
        else:
            np.savez(writer, **dump, allow_pickle=True)
        return writer.getvalue()


root = "/s1_md0/leiji/v-zhijian/MAD_data" # Your downloaded feature root
filenames = ["CLIP_frames_features_5fps"]  # "CLIP_language_tokens_features", "CLIP_language_features_MAD_test"
file = "CLIP_frames_features_5fps"
filename = os.path.join(root, file)
data = h5py.File("%s.h5" % filename, 'r')
feature_save_path = os.path.join(root, file)
output_clip_env = lmdb.open(feature_save_path, map_size=1099511627776)
for key, value in tqdm.tqdm(data.items()):
    with output_clip_env.begin(write=True) as output_txn:
        features_dict = {"features": np.array(value).astype(np.float32)}
        feature_dump = dumps_npz(features_dict, compress=True)
        output_txn.put(key=key.encode(), value=feature_dump)
