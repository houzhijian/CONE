##  Ego4D-NLQ leaderboard submission

### Highlight: improvement points compared with original paper 
* Token Feature Enhancement
* Multiscale window size during training
* Ensemble

###  Token Feature Enhancement
Better token feature leads to 1) more precise proposal boundary and 2) more discriminative proposal score.
We replace EgoVLP token feature with CLIP token feature or RoBERTa token feature

Please read the [Feature_Extraction_MD](./feature_extraction/README.md) to know how to extract CLIP or RoBERTa token features.

###  Multiscale window size during training
During training, we adopt multi-scale window size strategy as a data augmentation trick.
Please replace [original_dataloader_file](./cone/ego4d_mad_dataloader.py) with 
[new_dataloader_file](./cone/ego4d_dataloader_for_eccv2022_workshop.py) in the [cone/train.py](cone/train.py) and [cone/inference.py](cone/inference.py).


### Training
We train three model variants, each with different token features.

The actual command used for the CLIP token feature is
```
bash cone/scripts/train_ego4d_clip_leaderboard.sh 0 5 90 linear 
```


### Inference and ensemble
Then we perform inference for each model,
and finally ensemble three predictions by the following code for leaderboard submission
```
python ensemble.py
```