## ASVspoof5 Baseline

## Installation

First, downdoload the code through git clone.
```
git clone https://github.com/asvspoof-challenge/asvspoof5.git
```

To set up a new conda environment to run an experiment using GPU, follow the code below.
If you want to use your existing environment, please check the Pytorch and Cuda versions and run the last line only.

```
conda create --name aasist_baseline python=3.9
conda activate aasist_baseline
conda install pytorch==1.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Train / Test
Before running the experiment, replace the data directory of `database_path' in the config file of `./config/AASIST_ASVspoof5.conf`.

To train the model:
```
python ./main.py --config ./config/AASIST_ASVspoof5.conf
```

To evaluate the saved model:
```
python ./main.py --config ./config/AASIST_ASVspoof5.conf --eval
```
