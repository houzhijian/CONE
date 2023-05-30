## Requirements

This code requires some additional dependencies
```bash
pip install pyyaml easydict
```




## Training
Use the following commands for training:
```
python moment_localization/train.py --cfg experiments/ego4d/2D-TAN-64x64-K9L4-pool-sw-0.5bias-nms-con-match-adapt.yaml --verbose
```
```
python moment_localization/train.py --cfg experiments/mad/2D-TAN-64x64-K9L4-pool-sw-0.5bias-nms-con-match.yaml --verbose
```


## Testing

Then, run the following commands for evaluation: 
```
python moment_localization/test.py --cfg experiments/ego4d/2D-TAN-64x64-K9L4-pool-sw-0.5bias-nms-con-match-adapt.yaml --verbose --split val
python moment_localization/test.py --cfg experiments/ego4d/2D-TAN-64x64-K9L4-pool-sw-0.5bias-nms-con-match-adapt.yaml --verbose --split test
```
```
python moment_localization/test.py --cfg experiments/mad/2D-TAN-64x64-K9L4-pool-sw-0.5bias-nms-con-match.yaml --verbose --split val
python moment_localization/test.py --cfg experiments/mad/2D-TAN-64x64-K9L4-pool-sw-0.5bias-nms-con-match.yaml --verbose --split test
```


