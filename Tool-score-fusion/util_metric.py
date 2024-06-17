#!/usr/bin/env python
"""
Functions for evaluation

They are imported from eval_metrics.py in official t-DCF computatio package 
https://www.asvspoof.org/resources/tDCF_python_v2.zip

"""

from __future__ import print_function

import os
import sys
import numpy as np

__author__ = "ASVspoof committee"
__email__ = "www.asvspoof.org"
__copyright__ = "Copyright 2024, ASVspoof committee"

def compute_det_curve(target_scores, nontarget_scores):
    """ 
    frr, far, thr = compute_det_curve(target_scores, nontarget_scores)
    
    input
    -----
      target_scores:    np.array, target trial scores
      nontarget_scores: np.array, nontarget trial scores 

    output
    ------
      frr:   np.array, FRR, (#N, ), where #N is total number of scores + 1
      far:   np.array, FAR, (#N, ), where #N is total number of scores + 1
      thr:   np.array, threshold, (#N, )
    
    """
    
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), 
                             np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = (nontarget_scores.size - 
                            (np.arange(1, n_scores + 1) - tar_trial_sums))

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums/target_scores.size))
    # false rejection rates
    far = np.concatenate((np.atleast_1d(1), 
                          nontarget_trial_sums / nontarget_scores.size))  
    # false acceptance rates
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), 
                                 all_scores[indices]))  
    # Thresholds are the sorted scores
    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    """
    eer, eer_threshold = compute_eer(target_scores, nontarget_scores)
    
    input
    -----
      target_scores:    np.array, or list of np.array, target trial scores
      nontarget_scores: np.array, or list of np.array, nontarget trial scores 

    output
    ------
      eer:            float, EER 
      eer_threshold:  float, threshold corresponding to EER
    
    """
    if type(target_scores) is list and type(nontarget_scores) is list:
        frr, far, thr = compute_det_curve_sets(target_scores, nontarget_scores)
    else:
        frr, far, thr = compute_det_curve(target_scores, nontarget_scores)
    
    # find the operation point for EER
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)

    # compute EER
    eer = np.mean((frr[min_index], far[min_index]))    
    return eer, thr[min_index]


def compute_sasv_eer(scores, labels, tar_label = 1, nontar_label = 2, spoof_label = 0):
    """save_eer = compute_sasv_eer(scores, labels)
    
    input
    -----
      scores:    np.array, (N, ), fused scores for SASV
      labels:    np.array, (N, ), labels
      tar_label: int, label value for target bonafide data
      nontar_label: int, label value for nontarget bonafide data 
      spoof_label:  int, label value for spoofed data

    output
    ------
      save_eer:  float, SASV-EER 
    """
    tar_scores = scores[np.where(labels == tar_label)]
    other_scores = scores[np.where(labels != tar_label)]
    nontar_scores = scores[np.where(labels == nontar_label)]
    spoof_scores = scores[np.where(labels == spoof_label)]
        
    sasv_eer, sasv_thr = compute_eer(tar_scores, other_scores)
    return sasv_eer, sasv_thr

if __name__ == "__main__":
    print(__doc__)
