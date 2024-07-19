import os
import sys
from pathlib import Path
import pandas as pd


# column tags used in data frame
# trial name
g_filename_tag = 'filename'
# target speaker name
g_tar_speaker_tag = 'spk'
# asv label
g_asv_label_tag = 'asv_label'
# cm label
g_cm_label_tag = 'cm_label'
# asv score
g_asv_score_tag = 'asv-score'
# cm score
g_cm_score_tag = 'cm-score'
# sasv score
g_sasv_score_tag = 'sasv-score'
# attack
g_attack_tag = 'attack_anon'
# codec
g_codec_tag = 'codec'

# pooled condition
g_pooled_label = 'pooled'

# key labels
g_cm_bon = 'bonafide'
g_cm_spf = 'spoof'
g_asv_tar = 'target'
g_asv_non = 'nontarget'
g_asv_spf = 'spoof'

# for filtering out dummy labels
g_t2_tag = 'track_2_tag'
g_t2_dummy = 'dummy'

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
    return pd.read_csv(filepath, sep = sep, header=0).rename(
        columns={'tar_spk_anon': g_tar_speaker_tag,
                 'trial_anon': g_filename_tag,
                 'asv-label': g_asv_label_tag, 'cm-label': g_cm_label_tag})

def load_full_keys(filepath, sep=' '):
    """
    Load full key table

    input
    -----
      filepath: str, path to the tsv file
      sep: str, separator, default ' '

    output
    ------
      pd: pandas dataframe

    Assume tsv file, the first line is header
    """
    full_key_pd = load_tsv(filepath, sep)
    
    # manually change column names
    full_key_pd = full_key_pd.rename(
        columns={'tar_spk_anon': g_tar_speaker_tag, 'trial_anon': g_filename_tag,
                 'asv-label': g_asv_label_tag, 'cm-label': g_cm_label_tag})
    return full_key_pd

def load_cm_scores_keys(cm_scores_file, cm_keys_file, default_index=g_filename_tag):
    """
    input
    -----
      cm_scores_file: str, path to the CM score file

          filename	cm-score
          E_000001	0.01
          E_000002	0.02

      cm_keys_file: str, path to the CM key file
          filename	cm-label
          E_000001	bonafide
          E_000002	spoof

    output
    ------
      cm_scores: np.array, scores
      cm_keys: np.array, keys
    """
    assert cm_scores_file, "Please provide CM score file"
    assert cm_keys_file, "Please provide CM key file"
    assert Path(cm_scores_file).exists(), 'CM score file not exist'
    assert Path(cm_keys_file).exists(), 'CM key file not exist'

    # load tsv files
    cm_scores_pd = load_tsv(cm_scores_file).set_index(default_index)
    cm_keys_pd = load_tsv(cm_keys_file).set_index(default_index)

    assert set(cm_scores_pd.index) == set(cm_keys_pd.index), \
        'Error: CM score and key incompatible'

    # merge scores and keys
    cm_pd = cm_scores_pd.join(cm_keys_pd)
    
    return cm_pd[g_cm_score_tag].to_numpy(), cm_pd[g_cm_label_tag].to_numpy()


def load_sasv_scores_keys(sasv_scores_file, sasv_keys_file, default_index=[g_tar_speaker_tag, g_filename_tag]):
    """
    input
    -----
      sasv_scores_file: str, path to the SASV score file
         five columns
         spk        filename        cm-score          asv-score       sasv-score
         E_0101     E_0000000001    4.76273775100708  -1.0224495      1.87014412550354

      sasv_keys_file: str, path to the SASV key file
         spk        filename        cm-label          asv-label
         E_0101     E_0000000001    spoof             spoof

    output
    ------
      cm_scores: np.array, cm scores
      asv_scores:, np.array, asv_scores
      sasv_scores: np.array, sasv_scores
      cm_keys: np.array,
      asv_keys: np.array,
    """
    assert sasv_scores_file, "Please provide SASV score file"
    assert sasv_keys_file, "Please provide SASV key file"
    assert Path(sasv_scores_file).exists(), "SASV score file not exist"
    assert Path(sasv_keys_file).exists(), "SASV key file not exist"
    
    # load tsv files
    sasv_scores_pd = load_tsv(sasv_scores_file).set_index(default_index)
    sasv_keys_pd = load_tsv(sasv_keys_file).set_index(default_index)

    assert set(sasv_scores_pd.index) == set(sasv_keys_pd.index), \
        'Error: SASV score and key incompatible'
    
    # merge scores and keys
    sasv_pd = sasv_scores_pd.join(sasv_keys_pd)
    assert sasv_pd.shape[0] == sasv_scores_pd.shape[0], 'Error: SASV score and key incompatible'
    assert sasv_pd.shape[0] == sasv_keys_pd.shape[0], 'Error: SASV score and key incompatible'
    
    return sasv_pd[g_cm_score_tag].to_numpy(), \
        sasv_pd[g_asv_score_tag].to_numpy(), \
        sasv_pd[g_sasv_score_tag].to_numpy(), \
        sasv_pd[g_cm_label_tag].to_numpy(), \
        sasv_pd[g_asv_label_tag].to_numpy()



def load_cm_scores_keys_as_pd(cm_scores_file, cm_full_keys_file, default_index=g_filename_tag):
    """
    input
    -----
      cm_scores_file: str, path to the CM score file

          filename	cm-score
          E_000001	0.01
          E_000002	0.02

    output
    ------
      cm_score_pd: pd dataframe contains scores and keys
    """
    assert cm_scores_file, "Please provide CM score file"
    assert cm_full_keys_file, "Please provide CM key file"
    assert Path(cm_scores_file).exists(), 'CM score file not exist'
    assert Path(cm_full_keys_file).exists(), 'CM key file not exist'

    # load tsv files
    cm_scores_pd = load_tsv(cm_scores_file).set_index(default_index)
    cm_keys_pd = load_full_keys(cm_full_keys_file).set_index(default_index)

    # merge scores and keys
    cm_pd = cm_scores_pd.join(cm_keys_pd)

    assert set(cm_scores_pd.index) == set(cm_pd.index), \
        'Error: CM score and key incompatible'
    return cm_pd


def load_sasv_scores_keys_as_pd(sasv_scores_file, sasv_full_keys_file, default_index=[g_tar_speaker_tag, g_filename_tag]):
    """
    input
    -----
      sasv_scores_file: str, path to the SASV score file
         five columns
         spk        filename        cm-score          asv-score       sasv-score
         E_0101     E_0000000001    4.76273775100708  -1.0224495      1.87014412550354


    output
    ------
      sasv_score_pd: pd dataframe contains scores and keys
    """
    assert sasv_scores_file, "Please provide SASV score file"
    assert sasv_full_keys_file, "Please provide SASV key file"
    assert Path(sasv_scores_file).exists(), "SASV score file not exist"
    assert Path(sasv_full_keys_file).exists(), "SASV key file not exist"
    
    # load tsv files
    sasv_scores_pd = load_tsv(sasv_scores_file).set_index(default_index)
    sasv_keys_pd = load_full_keys(sasv_full_keys_file).set_index(default_index)

    # merge scores and keys
    sasv_pd = sasv_scores_pd.join(sasv_keys_pd)
    assert sasv_pd.shape[0] == sasv_scores_pd.shape[0], 'Error: SASV score and key incompatible'

    # filter out dummy data
    sasv_pd = sasv_pd.query('{:s} != "{:s}"'.format(g_t2_tag, g_t2_dummy))

    try:
        return sasv_pd[[g_asv_label_tag, g_cm_label_tag,
                        g_cm_score_tag, g_asv_score_tag, g_sasv_score_tag,
                        g_attack_tag, g_codec_tag]]
    except KeyError:
        return sasv_pd[[g_asv_label_tag, g_cm_label_tag,
                        g_asv_score_tag,
                        g_attack_tag, g_codec_tag]]



def return_attacks(data_key_pd, tag_attack_col=g_attack_tag):
    """
    return a list of spoofing attack labels 

    input
    -----
      data_key_pd: dataframe, contains both scores and full set of key files
      tag_attack_col: str, tag of the column for attack labels
    
    output
    ------
      attacks: list of str
    """
    # excldue the label bonafide 
    attacks = sorted(data_key_pd[tag_attack_col].unique())[:-1]
    return attacks

def return_codecs(data_key_pd, tag_codec_col=g_codec_tag):
    """
    return a list of codecs labels 

    input
    -----
      data_key_pd: dataframe, contains both scores and full set of key files
      tag_attack_col: str, tag of the column for attack labels
    
    output
    ------
      codecs: list of str
    """
    # excldue the label bonafide 
    codecs = sorted(data_key_pd[tag_codec_col].unique())
    return codecs

    
