t-EER: Parameter-Free Tandem Evaluation Metric of Countermeasures and Biometric Comparators
===============
This repository contains our implementation of the article published in IEEE Transactions on Pattern Analysis and Machine Intelligence (IEEE-T-PAMI), "t-EER: Parameter-Free Tandem Evaluation Metric of Countermeasures and Biometric Comparators". In this work we introduce a new metric for the joint evaluation of PAD solutions operating in situ with biometric verification.

[Paper link here](https://ieeexplore.ieee.org/abstract/document/10246406)

### t-EER paths + values
t-EER paths and t-EER values for different values of spoofing prevalence prior (ρ) on simulated scores drawn from three bivariate Gaussians. The overlaid blue curves on the lefthand side display the t-EER paths (one for each ρ), while the corresponding curves on the right-hand side display the corresponding t-EER value along each path. Three familiar special-case EERs are also indicated, along with proposed concurrent t-EER (magenta marker).
![Teer values using simulated scores](https://github.com/TakHemlata/T-EER/assets/44014715/58b61e4c-73c4-4890-8e1c-293228db0412)


### Python notebook
Link to run the notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ga7eiKFP11wOFMuZjThLJlkBcwEG6_4m?usp=sharing)

### Score file preparation
Set to use either synthetic, artificial scores, or upload real scores file containining separate countermeasure (CM) and automatic speaker verification (ASV) txt score files.

1. Upload CM and ASV scores file for experiments on SASV database.

   * Prepare a score file in a plain text format
```sh
LA_0015 LA_E_1103494 bonafide target 1.0000
LA_0015 LA_E_4861467 bonafide target 1.0000
...
```

2. Upload CM and ASV scores file for experiments on ASVspoof2021 LA database.
   (Keys and metadata are available [here](https://www.asvspoof.org/))

   * Prepare a score file in a plain text format
   
```sh
CM score file:
LA_0009 LA_E_9332881 alaw ita_tx A07 spoof notrim eval -5.8546
LA_0020 LA_E_7294490 g722 loc_tx bonafide bonafide notrim eval -5.8546
...

ASV score file:
LA_0007-alaw-ita_tx  LA_E_5013670-alaw-ita_tx  alaw  ita_tx  bonafide  nontarget  notrim  eval -4.8546
LA_0008-alaw-ita_tx  LA_E_5013671-alaw-ita_tx  alaw  ita_tx  bonafide target  notrim  eval -4.8546
...
```

### To run the script:
The Boolean argument depends on the baseline you wish to use, either "SASV" or "ASVspoof 2021".
```
python evaluate_tEER.py true false
```

Result for SASV baseline:
t-EER = 2.28 %

### Contact
For any query regarding this repository, please contact:

- Hemlata Tak: tak[at]eurecom[dot]fr
- Tomi H. Kinnunen: tomi.kinnunen[at]uef[dot]fi

## Citation
If you use this metric in your work then use the following citation:

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
