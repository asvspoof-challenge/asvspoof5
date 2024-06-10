# AASIST ASVspoof5 Baseline
By Hye-jin Shim, Carnegie Mellon University, 2024


## Requirement

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

## Usage
Before running the experiment, replace the data directory of `database_path` in the config file of `./config/AASIST_ASVspoof5.conf`.

To train the model:
```
python ./main.py --config ./config/AASIST_ASVspoof5.conf
```

To evaluate the saved model (showed `EER: 15.2%` on validation set):
* Evaluation only phase considers both of Phase 1 and Phase 2 evaluation metrics
```
python ./main.py --config ./config/AASIST_ASVspoof5.conf --eval
```

## Citation
This code is based on https://github.com/clovaai/aasist. If you use AASIST model, please cite the following paper:
```
@inproceedings{jung2022aasist,
  title={Aasist: Audio anti-spoofing using integrated spectro-temporal graph attention networks},
  author={Jung, Jee-weon and Heo, Hee-Soo and Tak, Hemlata and Shim, Hye-jin and Chung, Joon Son and Lee, Bong-Jin and Yu, Ha-Jin and Evans, Nicholas},
  booktitle={ICASSP 2022-2022 IEEE international conference on acoustics, speech and signal processing (ICASSP)},
  pages={6367--6371},
  year={2022},
  organization={IEEE}
}
```
