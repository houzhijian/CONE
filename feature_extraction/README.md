## How to extract visual and textual feature 

### Ego4D-NLQ 

Request access to the [Epo4D](https://ego4d-data.org/docs/start-here/) dataset.

We use [EgoVLP](https://github.com/showlab/EgoVLP) to extract visual and textual features for the Ego4D-NLQ dataset.

The EgoVLP authors provide codes on how to extract both visual, textual query and textual token features.
Please refer to [EgoVLP_extractor](../run_on_video/egovlp_extrator.py) or run/test_nlq.py in [EgoVLP](https://github.com/showlab/EgoVLP).

In additional, we find that the performance empirically increases when the textual token feature extractor is replaced by
CLIP or RoBERTa.

We also provide the code to extract the token feature by CLIP or RoBERTa.

```
python ego4d_clip_token_extractor.py 
python ego4d_roberta_token_extractor.py --train_data_file JSON_DATA_PATH  \
      --do_extract --token_fea_dir SAVE_PATH
```

Then merge the textual query and token feature into a single LMDB file.

```
python ego4d_merge_textual_cls_token_feature.py 
```

### MAD
Request access to the [MAD](https://github.com/Soldelli/MAD) dataset.

The MAD authors claim that due to copyright constraints, MAD’s videos will not be publicly released.
We use their extracted CLIP visual feature and convert official h5 file to the lmdb file for our further processing.

```
python convert_h5_to_lmdb.py 
```

We extract CLIP textual query and token features through the following code.
```
python mad_clip_text_extractor.py 
```

### Additional information
We recommend this repo [HERO_Video_Feature_Extractor](https://github.com/linjieli222/HERO_Video_Feature_Extractor) if you want to extract other video features (e.g., SlowFast, ResNet).

