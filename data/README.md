## File Pre-processing


This section hosts the detail information to conduct file pre-processing for both dataset.


### Simple Action
Please copy and paste the files inside the ``data`` directory of our released Ego4D-NLQ and MAD data to the current directory.



### FILE SUMMARY
We have provided the pre-processed files (inside the ``data/ego4d_data`` or ``data/mad_data`` directory).

Each file is in [JSON Line](https://jsonlines.org/) format, each row of the files can be loaded as a single `dict` in Python. Below is an example of the annotation:

```
{
    "query": "what did I put in the black dustbin?", 
    "query_id": ca7e11a2-cd1e-40dd-9d2f-ea810ab6a99b_0, 
    "duration": 480, 
    "clip_id": "93231c7e-1cf4-4a20-b1f8-9cc9428915b2", 
    "video_id": "38737402-19bd-4689-9e74-3af391b15feb",
    "timestamps": [425.0, 431.0], 
}
```
`query_id` is a unique identifier of a `query`. This query corresponds to a video identified by its id `clip_id`. 

`duration` is an integer indicating the duration of this video.

`clip_id` is a unique identifier of a video clip. For Ego4D benchmark, each video clip is actually trimmed from a full-length video identified by its id `video_id`. 
Nevertheless, in our long video grounding problem setting, the video clip is reckoned as a long-form video.

`timestamps` is the ground-truth moment and a list with two scalars, the first scalar is ground-truth begin timestamp, the second scalar is groundtruth end timestamp.

Below shows the detailed procedures for file pre-processing, if you are interested.


### Ego4D-NLQ 
First get official json files (we also provide them in ``data/ego4d_ori_data`` directory under our released Ego4D-NLQ data). Then pre-process them through the following codes.
```
python reformat_data.py --input_train_split ego4d_ori_data/nlq_train.json --input_val_split ego4d_ori_data/nlq_val.json 
--input_test_split ego4d_ori_data/nlq_test_unannotated.json --output_save_path ego4d_data --dset_name ego4d

python process_train_split.py --_train_split ego4d_data/train.jsonl  --dset_name ego4d
```

The first code aims to convert the original released file to our standard jsonl file for further processing, 
and the second code aims to  remove some low-quality train samples. 

### MAD
First get official json files (we also provide them in ``data/mad_ori_data`` directory under our released MAD data). Then pre-process them through the following codes.
```
python reformat_data.py --input_train_split mad_ori_data/MAD_train.json --input_val_split mad_ori_data/MAD_val.json 
--input_test_split mad_ori_data/MAD_test.json --output_save_path mad_data --dset_name mad

python process_train_split.py --train_split mad_data/train.jsonl  --dset_name mad
```