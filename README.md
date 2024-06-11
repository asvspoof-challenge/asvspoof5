# Asvspoof5 Baselines 

By [ASVspoof5 challenge organizers](https://www.asvspoof.org/)

## Baseline CMs (track 1)

Three baselines for CMs are available: 

* Baseline-RawNet2 (PyTorch) <br/> End-to-End DNN classifier
* Baseline-AASIST (PyTorch) <br/> End-to-End graph attention-based classifier


## Baseline SASV (track 2)

Two baselines for SASV are available: 

* SASV Fusion-based baseline from SASV 2022 challenge [here](https://github.com/sasv-challenge/SASVC2022_Baseline)
* Single integrated SASV baseline: [here](https://github.com/sasv-challenge/SASV2_Baseline/tree/asvspoof5)
  * This is an adapted version of a work previously introduced in Interspeech 2023. Use the above link to access the `asvspoof5` branch.
  * Download the code directly: https://github.com/sasv-challenge/SASV2_Baseline/archive/refs/tags/ASVspoof.v0.0.1.tar.gz

## Evaluation metrics

Track 1
* minDCF (primary)
* CLLR, EER (secondary)

Track 2
* a-DCF (primary)
* min t-DCF and t-EER (secondary)
