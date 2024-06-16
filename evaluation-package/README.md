# ASVspoof5 Evaluation Package
We provide this evaluation package to compute the evaluation metrics for both Phase 1 and Phase 2.

Track 1
* min DCF (primary)
* CLLR, EER (secondary)

Track 2
* a-tdcf (primary)
* min t-DCF and t-EER (secondary)


## Usage
For the Track 1, cm score 

1. Track 1
Calucate minDCF, CLLR, and EER by giving one cm score file
```
python evaluation.py -m p1 -cm cm_score_file
```

2. Track 2
Calucate a-DCF, min t-DCF, and t-EER by giving cm score file (and asv score file for min t-DCF and t-EER)
To compute a-DCF, `pip install a_dcf` will install the package.
For further details, please check [([Github](https://github.com/shimhz/a_DCF))].
```
# To calculate all metrics, 
python evaluation.py -m p2_tandem -cm cm_score_file -asv asv_score_file

# To calculate a-DCF only, 
python evaluation.py -m p2_single -cm cm_score_file
```

### Score file format
- The score file format should include four columns:
  - (i) speaker model, (ii) test utterance, (iii) score, and (iv) trial type
- Example of score file
```
# CM score file (for minDCF, CLLR, EER, and a-DCF)
# <speaker_id> <utterance_id> <score> <trial type> 
LA_0015 LA_E_1103494 6.960134565830231 bonafide
LA_0007 LA_E_5013670 6.150891035795212 bonafide
LA_0007 LA_E_7417804 -2.306972861289978 spoof

# ASV score file (for min t-DCF and t-EER)
# <cm_label> <sasv_label> <score>
bonafide target 36.78691
bonafide nontarget -39.15536
A01 spoof 27.07672
```


## Citation
If you use provided evaluation metrics, please cite the following papers:

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
