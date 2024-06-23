import os
import sys
import pandas as pd

def load_tsv(filepath, sep='\t'):
    """
    input
    -----
      filepath: str, path to the tsv file
      sep: str, separator, default \t

    output
    ------
      pd: pandas dataframe

    Assume tsv file, the first line is header
    """
    return pd.read_csv(filepath, sep = sep, header=0)

def load_cm_scores_keys(cm_scores_file, cm_labels_file, default_index='filename'):
    """
    input
    -----
      cm_scores_file: str, path to the CM score file
      cm_keys_file: str, path to the CM key file

    output
    ------
      cm_scores: np.array, scores
      cm_keys: np.array, keys
    """

    # load tsv files
    cm_scores_pd = load_tsv(cm_scores_file).set_index(default_index)
    cm_keys_pd = load_tsv(cm_labels_file).set_index(default_index)

    assert set(cm_scores_pd.index) == set(cm_keys_pd.index), \
        'Error: CM score and key incompatible'
    
    # merge scores and keys
    cm_pd = cm_scores_pd.join(cm_keys_pd)
    
    return cm_pd['cm-score'].to_numpy(), cm_pd['cm-label'].to_numpy()


def load_sasv_scores_keys(sasv_scores_file, sasv_keys_file, default_index='filename'):
    """
    input
    -----
      sasv_scores_file: str, path to the SASV score file
         four columns
         filename        cm-score        asv-score       sasv-score
         E_0000103882    4.76273775100708        -1.0224495      1.87014412550354

      sasv_keys_file: str, path to the SASV key file
         filename        cm-label        asv-label
         E_0007976787    spoof   spoof

    output
    ------
      cm_scores: np.array, cm scores
      asv_scores:, np.array, asv_scores
      sasv_scores: np.array, sasv_scores
      cm_keys: np.array,
      asv_keys: np.array,
    """
    # load tsv files
    sasv_scores_pd = load_tsv(sasv_scores_file).set_index(default_index)
    sasv_keys_pd = load_tsv(sasv_keys_file).set_index(default_index)

    assert set(sasv_scores_pd.index) == set(sasv_keys_pd.index), \
        'Error: SASV score and key incompatible'
    
    # merge scores and keys
    sasv_pd = sasv_scores_pd.join(sasv_keys_pd)
    
    return sasv_pd['cm-score'].to_numpy(), sasv_pd['asv-score'].to_numpy(), \
        sasv_pd['sasv-score'].to_numpy(), \
        sasv_pd['cm-label'].to_numpy(), sasv_pd['asv-label'].to_numpy()
