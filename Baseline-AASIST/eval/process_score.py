import numpy as np
import sys

def process_sasv_score_files(cm_score_file, asv_score_file):
    cm_data = np.genfromtxt(cm_score_file, dtype=str)
    cm_utt_id = cm_data[:,1]
    cm_trial_type = cm_data[:, 3]
    cm_scores = cm_data[:,4].astype(float)

    asv_data = np.genfromtxt(asv_score_file, dtype=str)
    asv_utt_id = asv_data[:,1]
    asv_trial_type = asv_data[:, 3]
    asv_scores = asv_data[:,4].astype(float)
    
    # need to check here that the utt_ids are the same in cm and asv files
    X_tar = np.array([asv_scores[asv_trial_type == 'target'], cm_scores[cm_trial_type == 'target']]).T
    X_non = np.array([asv_scores[asv_trial_type == 'nontarget'], cm_scores[cm_trial_type == 'nontarget']]).T
    X_spf = np.array([asv_scores[asv_trial_type == 'spoof'], cm_scores[cm_trial_type == 'spoof']]).T

    return X_tar, X_non, X_spf

def process_ASVspoof2021_score_files(cm_score_file, asv_score_file,cond):
    cm_data = np.genfromtxt(cm_score_file, dtype=str)
    cm_utt_id = cm_data[:,1]
    cm_trial_type = cm_data[:, 5]
    cm_cond = cm_data[:,7]
    cm_scores = cm_data[:,8].astype(float)
    cm_scores = cm_scores[cm_cond == cond]
    cm_trial_type = cm_trial_type[cm_cond == cond]

    asv_data = np.genfromtxt(asv_score_file, dtype=str)
    asv_utt_id = asv_data[:,1]
    asv_trial_type = asv_data[:, 5]
    asv_cond = asv_data[:,7]
    asv_scores = asv_data[:,8].astype(float)
    asv_trial_type = asv_trial_type[asv_cond == cond]
    asv_scores = asv_scores[asv_cond == cond]

    X_tar = np.array([asv_scores[asv_trial_type == 'target'], cm_scores[cm_trial_type == 'bonafide']],dtype=object).T
    X_non = np.array([asv_scores[asv_trial_type == 'nontarget'], cm_scores[cm_trial_type == 'nontarget']],dtype=object).T
    X_spf = np.array([asv_scores[asv_trial_type == 'spoof'], cm_scores[cm_trial_type == 'spoof']],dtype=object).T

    return X_tar, X_non, X_spf