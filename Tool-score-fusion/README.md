# ASV and CM score fusion package

This package can be used to fuse ASV and CM scores in Track 2.

The idea is to calibrate the ASV and CM scores into log-likelihood-ratios-like statisics, then fuse them based on a non-linear fuction. See more in `Reference`.

## Dependency

```bash
scikt-learn
pandas
numpy
```

The latest version should work.  Otherwise, try to install packages listed in `requirements.txt`

## Usage


```bash
python score-fusion.py --dev_input_csv <dev_input_csv> --eval_input_csv <eval_input_csv> --eval_output_csv <eval_output_csv>
```


**Input**: dev_input_csv

A CSV file containing ASV and CM scores and labels of development set data.

```bash
asv_score,cm_score,sasv_label
0.123,0.245,1.0
0.132,0.354,0.0
...
```

* The first column `asv_score` is the raw ASV system score
* The second column `cm_score` is the raw CM system score
* The third column `sasv_label` is the SASV (ASV) label
    * 1.0: target bona fide trial
    * 2.0: non-target bona fide trial
    * 3.0: spoofed trial

**Input**: eval_input_csv

A CSV file containing ASV and CM scores of evaluation set data. Same format as development set data but without `sasv_label`

**Output**: eval_output_csv

A CSV file containing ASV, CM, and fused scores for evaluation set data.
```bash
asv_score,cm_score,fused_score
0.123,0.245,0.923
0.132,0.354,0.872
...
```

## Demo

```bash
python score-fusion.py 
```

This will use the example files `dev_scores_label.csv` and `eval_scores.csv` as input and produce `eval_scores_fused.csv`.

## Reference

The code is from https://github.com/nii-yamagishilab/SpeechSPC-mini

A tutorial notebook on the fusion strategy is available [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1D9YZLkSTwXkZGnZAtLpl-1w9ZG2hUxOY?usp=sharing)

The technical details are described in the paper below

```bibtex
@inproceedings{wangRevisiting2024,
  title = {Revisiting and {{Improving Scoring Fusion}} for {{Spoofing-aware Speaker Verification Using Compositional Data Analysis}}},
  booktitle = {Proc. {{Interspeech}}},
  author = {Wang, Xin and Kinnunen, Tomi and Kong Aik, Lee and Noe, Paul-Gauthier and Yamagishi, Junichi},
  year = {2024},
  pages = {(accepted)}
}
```

## LICENSE

```
Copyright 2024 Wang Xin

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---
This work is supported by JST PRESTO Grant Number JPMJPR23P9, Japan.
