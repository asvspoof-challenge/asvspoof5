#!/usr/bin/env python3
"""
Adopted from https://github.com/shimhz/a_DCF

a-DCF: an architecture agnostic metric with application to 
spoofing-robust speaker verification, published in Odyssey 2024.
"""

import sys
import os
from typing import List
from dataclasses import dataclass

import numpy as np


class CostModel:
    """Class describing SASV-DCF's relevant costs"""
    def __init__(self, Pspf = 0.05, Pnontrg = 0.05, Ptrg = 0.9,
                 Cmiss = 1, Cfa_asv = 10, Cfa_cm = 20):
        self.Pspf = Pspf
        self.Pnontrg = Pnontrg
        self.Ptrg = Ptrg
        self.Cmiss = Cmiss
        self.Cfa_asv = Cfa_asv
        self.Cfa_cm = Cfa_cm
        

def calculate_a_dcf(
    sasv_score_dir: str,
    cost_model: CostModel = CostModel(),
    printres: bool = True,
    ):

    data = np.genfromtxt(sasv_score_dir, dtype=str, delimiter=" ")
    scores = data[:, 2].astype(np.float64)
    keys = data[:, 3]

    # Extract target, nontarget, and spoof scores from the ASV scores
    trg = scores[keys == 'target']
    nontrg = scores[keys == 'nontarget']
    spf = scores[keys == 'spoof']

    return _calculate_a_dcf(trg, nontrg, spf, cost_model, printres)


def _calculate_a_dcf(
        trg,
        nontrg,
        spf,
        cost_model: CostModel = CostModel(),
        printres: bool = True,
    ):

    far_asvs, far_cms, frrs, a_dcf_thresh = compute_a_det_curve(trg, nontrg, spf)

    a_dcfs = np.array([cost_model.Cmiss * cost_model.Ptrg]) * np.array(frrs) + \
        np.array([cost_model.Cfa_asv * cost_model.Pnontrg]) * np.array(far_asvs) + \
        np.array([cost_model.Cfa_cm * cost_model.Pspf]) * np.array(far_cms)

    a_dcfs_normed = normalize(a_dcfs, cost_model)

    min_a_dcf_idx = np.argmin(a_dcfs_normed)
    min_a_dcf = a_dcfs_normed[min_a_dcf_idx]
    min_a_dcf_thresh = a_dcf_thresh[min_a_dcf_idx]
    x_axis = np.arange(len(a_dcfs_normed))

    dcf_msg = f"a-DCF: {min_a_dcf:.5f}, threshold: {min_a_dcf_thresh:.5f}"

    if printres:
        print(dcf_msg)
        print(cost_model)

    return {
        "min_a_dcf": min_a_dcf,
        "min_a_dcf_thresh": min_a_dcf_thresh,
    }


def normalize(a_dcfs: np.ndarray, cost_model: CostModel) -> np.ndarray:
    a_dcf_all_accept = np.array([cost_model.Cfa_asv * cost_model.Pnontrg + \
        cost_model.Cfa_cm * cost_model.Pspf])
    a_dcf_all_reject = np.array([cost_model.Cmiss * cost_model.Ptrg])

    a_dcfs_normed = a_dcfs / min(a_dcf_all_accept, a_dcf_all_reject)

    return a_dcfs_normed


def compute_a_det_curve(trg_scores: np.ndarray, nontrg_scores: np.ndarray, spf_scores: np.ndarray) -> List[List]:

    all_scores = np.concatenate((trg_scores, nontrg_scores, spf_scores))
    labels = np.concatenate(
        (np.ones_like(trg_scores), np.zeros_like(nontrg_scores), np.ones_like(spf_scores) + 1))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]
    scores_sorted = all_scores[indices]

    fp_nontrg, fp_spf, fn = len(nontrg_scores), len(spf_scores), 0
    far_asvs, far_cms, frrs, a_dcf_thresh = [1.], [1.], [0.], [float(np.min(scores_sorted))-1e-8]
    for sco, lab in zip(scores_sorted, labels):
        if lab == 0: # non-target
            fp_nontrg -= 1 # false alarm for accepting nontarget trial
        elif lab == 1: # target
            fn += 1 # miss
        elif lab == 2: # spoof
            fp_spf -= 1 # false alarm for accepting spof trial
        else:
            raise ValueError ("Label should be one of (0, 1, 2).")
        far_asvs.append(fp_nontrg / len(nontrg_scores))
        far_cms.append(fp_spf / len(spf_scores))
        frrs.append(fn / len(trg_scores))
        a_dcf_thresh.append(sco)

    return far_asvs, far_cms, frrs, a_dcf_thresh


if __name__ == "__main__":
    # default a-dcf cost func
    if len(sys.argv) == 2:
        sys.exit(
            calculate_a_dcf_eers(
                sys.argv[1])
        )
    else:
        costmodel = CostModel(
            Pspf=float(sys.argv[2]),
            Pnontrg=float(sys.argv[3]),
            Ptrg=float(sys.argv[4]),
            Cmiss=float(sys.argv[5]),
            Cfa_asv=float(sys.argv[6]),
            Cfa_cm=float(sys.argv[7]),
        )
        sys.exit(calculate_a_dcf_eers(sys.argv[1], cost_model = costmodel))
