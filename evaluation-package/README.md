# ASVspoof5 Evaluation Package
We provide this evaluation package to compute the evaluation metrics for both Phase 1 and Phase 2.

Track 1
* min DCF (primary)
* CLLR, EER, actDCF (secondary)

Track 2
* a-tdcf (primary)
* min t-DCF and t-EER (secondary)


## Requirements

Scipy, numpy, and pandas. No specific requirement on the version.

If using conda, you can install an environment by
```bash
conda create --name <ENV_NAME> python=3.8.0 scipy=1.10.1 pandas=1.2.4
```


## Usage
For the Track 1, cm score 

1. Track 1
   
Calculate minDCF, CLLR, and EER by giving one cm score file and key file
```
python evaluation.py --m t1 --cm cm_score_file --cm_key cm_key_file
```

2. Track 2

Calculate a-DCF, min t-DCF, and t-EER by giving an SASV score file and key file

```
# To calculate all metrics, 
python evaluation.py --m t2_tandem --sasv sasv_score_file --sasv_key sasv_key_file

# To calculate a-DCF only, 
python evaluation.py --m t2_single --sasv sasv_score_file --sasv_key sasv_key_file
```

### Score file format

cm_score_file must have two columns, separated by \t, with header

```bash
filename	cm-score
E_000001	0.01
E_000002	0.02
```

cm_key_file must have two columns, separated by \t, with header
```bash
filename	cm-label
E_000001	bonafide
E_000002	spoof
```

sasv_score_file must have four columns, separated by \t, with header
```bash
filename	cm-score	asv-score	sasv-score
E_000001	0.01		0.01		0.04
E_000002	0.02		0.02		0.05
```

if sasv system produces no cm-score and asv-score
```bash
filename	cm-score	asv-score	sasv-score
E_000001	-		-		0.04
E_000002	-		-		0.05
```

sasv_key_file must have three columns, separated by \t, with header
```bash
filename	cm-label	asv-label
E_000001	bonafide	target
E_000002	spoof		spoof
E_000003	bonafide	nontarget
```


## Citation
If you use the provided evaluation metrics, please cite the following papers:

* a-DCF
```bibtex
@inproceedings{shim2024adcf,
  title={{a-DCF}: an architecture agnostic metric with application to spoofing-robust speaker verification},
  author={Hye-jin Shim and Jee-weon Jung and Tomi Kinnunen and others},
  year={2024},
  booktitle={Proc. Speaker Odyssey},
  note={To appear},
  eprint={2403.01355},
  archivePrefix={arXiv},
  primaryClass={eess.AS}
}
```

* t-DCF
```bibtex
@ARTICLE{9143410,
  author={Kinnunen, Tomi and Delgado, Héctor and Evans, Nicholas and Lee, Kong Aik and Vestman, Ville and Nautsch, Andreas and Todisco, Massimiliano and Wang, Xin and Sahidullah, Md and Yamagishi, Junichi and Reynolds, Douglas A.},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={Tandem Assessment of Spoofing Countermeasures and Automatic Speaker Verification: Fundamentals}, 
  year={2020},
  volume={28},
  number={},
  pages={2195-2210},
  keywords={Error analysis;Measurement;Cost function;Speech processing;Electronic mail;Security;IEEE transactions;Automatic speaker verification (ASV);detect- ion cost function;presentation attack detection;spoofing counter- measures},
  doi={10.1109/TASLP.2020.3009494}}
```

* t-EER
```bibtex
@ARTICLE {Kinnunen2023-tEER,
author = {T. H. Kinnunen and K. Lee and H. Tak and N. Evans and A. Nautsch},
journal = {{IEEE} Transactions on Pattern Analysis and Machine Intelligence},
title = {t-EER: Parameter-Free Tandem Evaluation of Countermeasures and
Biometric Comparators (to appear)},
doi = {10.1109/TPAMI.2023.3313648},
year = {2023},
publisher = {IEEE Computer Society},
}
```
