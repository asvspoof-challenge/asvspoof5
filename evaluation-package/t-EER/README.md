t-EER secondary metric for track 2: 
### Score file preparation
upload scores file containining separate countermeasure (CM) and automatic speaker verification (ASV) txt score files.

1. Upload CM and ASV scores file

   * Prepare a ASV and CM score file in a plain text format
```sh
E_0062 D_0000000001 1.0000
E_0063 D_0000000002 1.0000
...
```

2. Upload the groundtruth file
```sh
target
nontarget
spoof
....
``` 
 
### To run the script:
```
python Compute_tEER.py ASV_file_path CM_file_path ground_truths
```

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
