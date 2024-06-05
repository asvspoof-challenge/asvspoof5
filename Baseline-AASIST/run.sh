#!/bin/bash

# Training
python ./main.py --config ./config/AASIST_ASVspoof5.conf

# Evaluation
# python ./main.py --config ./config/AASIST_ASVspoof5.conf --eval