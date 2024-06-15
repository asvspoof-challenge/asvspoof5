# RawNet2 ASVspoof5 baseline

By Hemlata Tak
The code in this repository serves as one of the baselines of the ASVspoof5 challenge, using an end-to-end method that uses a model based on the RawNet2 topology as described [here](https://arxiv.org/abs/2011.01108).

## Installation
First, clone the repository locally, create and activate a conda environment, and install the requirements :
```
$ git clone https://github.com/asvspoof-challenge/asvspoof5.git
$ cd asvspoof5/Baseline-RawNet2/
$ conda create --name rawnet2_baseline python=3.7
$ conda activate rawnet2_baseline
$ conda install pytorch=1.8.0 -c pytorch
$ pip install -r requirements.txt
```

### Training
To train the model run:
```
python Main_RawNet2_baseline.py
```

### Testing
```
python Main_RawNet2_baseline.py --is_eval --eval --model_path='/path/to/your/your_best_model.pth' --eval_output='eval_CM_scores.txt'
```
  
