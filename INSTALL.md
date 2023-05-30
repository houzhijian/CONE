# Requirements

This code requires Python, PyTorch, and a few other Python libraries. 
We recommend creating conda environment and installing all the dependencies, as follows:

conda install
- python 3.8+
- pytorch 
- numpy 
- tqdm
- tensorboard
- pandas
- scipy

pip install
- lmdb
- terminaltables
- tabulate

An example could be look like the below:
```bash
conda create -n cone python=3.8
conda activate cone
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install numpy tqdm tensorboard pandas scipy
pip install lmdb terminaltables tabulate
pip install msgpack-numpy msgpack
```