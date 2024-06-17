#!/usr/bin/env python
"""
Score fusion of ASV and CM based on calibrated LLRs.

Usage:

"""

import os
import sys
import numpy as np
import pandas as pd
import argparse
import sklearn.linear_model
from sklearn.mixture import GaussianMixture

import util_metric

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2024, Xin Wang"

###
# argparse
###

parser = argparse.ArgumentParser(
    prog='Score-fusion',
    description='Fuse ASV and CM scores into SASV score',
    epilog='')

parser.add_argument('--dev_input_csv', type = str, 
                    default = 'dev_scores_label.csv',
                    help='input csv file of development set scores and labels')
parser.add_argument('--eval_input_csv', type = str, 
                    default = 'eval_scores.csv',
                    help='input csv file of evaluation set scores')
parser.add_argument('--eval_output_csv', type = str, 
                    default = 'eval_scores_fused.csv',
                    help='output csv file of evaluation set fused scores')
parser.add_argument('--label_value_target', type = int, 
                    default = 1,
                    help='label value for target bonafide data, default 1')
parser.add_argument('--label_value_nontarget', type = int, 
                    default = 2,
                    help='label value for nontarget bonafide data, default 2')
parser.add_argument('--label_value_spoof', type = int, 
                    default = 0,
                    help='label value for spoofed data, default 0')

parser.add_argument('--use_linear_fusion', action='store_true', 
                    default = False,
                    help='whether use linear rather than nonlinear fusion')

args = parser.parse_args()

###
# Functions to fuse LLRs
###

def fuse_llr_linear(llr_asv, llr_cm):
    """
    s_sasv = fuse_llr_linear(llr_cm, threshold)
    
    fusion by linearly combine LLRs

    input: llr_cm, scalar or np.array, llr of CM
    input: llr_asv, scalar or np.array, llr of ASV
    output: s_sasv, fused score for SASV 
    """
    return llr_asv + llr_cm

def fuse_llr_nonlinear(llr_asv, llr_cm, rho):
    """
    s_sasv = fuse_llr_nonlinear(llr_cm, threshold, rho)

    fusion by non-linearly combine LLRs

    input: llr_cm, scalar or np.array, llr of CM
    input: llr_asv, scalar or np.array, llr of ASV
    input: rho, scalar, spoofing prior
    output: s_sasv, fused score for SASV 
    """
    # log sum exp trick to prevent overflow
    # it does not affect the result
    shift = (llr_asv + llr_cm) / 2.0
    
    assert rho >= 0 and rho <= 1.0, "invalid rho"

    tmp = (1 - rho) * np.exp(-llr_asv + shift) + rho * np.exp(-llr_cm + shift)
    return - ( np.log(tmp) - shift )
    
###
# Generative calibration based on Gaussian
###
def fit_gaussians(data):
    """
    gm = fit_gaussians(data)

    input: data, np.array, (N, 2), data[:, 0] ASV score, data[:, 1] CM score
    output: gm, sklearn Gaussian model object
    """
    gm = GaussianMixture(n_components=1, random_state=0)
    gm.fit(data)
    return gm

def get_llrs(data, gm_tar, gm_non, gm_spf):
    """
    llrs = get_llrs(data, gm_tar, gm_non, gm_spf)
    
    compute the LLRs based on estimated Gaussian score distributions
    
    input: data, np.array(N, 2), data[:, 0] ASV score, data[:, 1] CM score
    input: gm_tar, sklearn Gaussian model for target (bonafide) data
    input: gm_non, sklearn Gaussian model for nontarget (bonafide) data
    input: gm_spf, sklearn Gaussian model for spoofed data
    
    output: llr, np.array(N, 2), s[:, 0] -> LLR_asv, s[:, 1] -> LLR_cm
    """
    # [LLR tar-non, LLR tar-spf]
    return np.stack(
        [gm_tar.score_samples(data) - gm_non.score_samples(data),
         gm_tar.score_samples(data) - gm_spf.score_samples(data)], axis=1)


###
# Logistic-regression-based calibration
###

def logistic_reg(input_data, labels, pos_prior):
    """
    scale, bias = logistic_reg(input_data, labels, pos_prior)
    
    input
    -----
      input_data: np.array, (N, ) or (N, 1), input data of pos and neg classes
      labels:     np.array, label of each datum, pos (1) or neg classes (0), 
      pos_prior:  scalar, prior ratio of positive class
    
    output
    ------
      scale:      scalar, scale parameer a in f(x) = ax+b
      bias:       scalar, bias parameer b in f(x) = ax+b
    """
    # get the prior weights for logistic regression
    prior_weight = {0: 1-pos_prior, 1:pos_prior}
    # we need to subtract the impact of prior on the bias
    prior_logit = np.log(pos_prior /(1-pos_prior))

    # fit the linear model
    reg_model = sklearn.linear_model.LogisticRegression(class_weight = prior_weight)

    # 
    if input_data.ndim == 1:
        reg_model.fit(np.expand_dims(input_data, axis=1), labels)
    else:
        reg_model.fit(input_data, labels)

    scale = reg_model.coef_[0]
    bias = reg_model.intercept_ - prior_logit
    return scale, bias

def get_calibration_affine(raw_scores, labels):
    """
    func = get_calibration_affine(raw_scores, labels)
    
    Get a function f(x) = ax+b that fits {raw_scores labels} for logistic reg.

    input
    -----
      raw_scores: np.array, (N, 1) or (N, ), input data x
      labels:     np.array, (N, ), data label,  1 and 0 for positive negative
    
    output
    ------
      func:       a lambda function, y = f(x) gives calibrated scores
    """
    # compute the prior of positive class
    pos_prior = np.sum(labels) / labels.shape[0]
    # get the scale and bias for ax+b
    scale, bias = logistic_reg(raw_scores, labels, pos_prior)
    return lambda x: x * scale + bias

###
# A wrapper over functions to compute SASV eer 
###
def compute_sasv_eer(scores, labels):
    """save_eer = compute_sasv_eer(scores, labels)
    
    input
    -----
      scores:    np.array, (N, ), fused scores for SASV
      labels:    np.array, (N, ), labels
    output
    ------
      save_eer:  float, SASV-EER 
    """
    return util_metric.compute_sasv_eer(
        scores, labels, args.label_value_target, args.label_value_nontarget,
        args.label_value_spoof)



###
# Grid search
###

def grid_search_rho(data, labels, get_metric = compute_sasv_eer):
    """
    rho = 
    input
    -----
      data:       np.array, (N, 2), data[:, 0] ASV score, data[:, 1] CM score
      labels:     np.array, (N, ), SASV label, 1 for target bonafide, 
                   2 for nontarget bonafide, 0 for spoofed

      get_metric: a function computes metric = func_metric(fused_scores, labels)
                  it measures how well the fused scores work
    output
    ------
      rho:     scalar, the spoofing prevelance prior used in non-linear fusion

    """
    # search for the best rho on the development set
    
    ### 
    # fixd parameters for grid search
    ###
    eer = 1.0
    rho_best = None
    step_num = 10
    iter_num = 5
    grid_start = 0.0
    grid_end = 1.0


    # for each iteration
    for iter_idx in np.arange(iter_num):
        # for each candicate value
        for rho in np.linspace(grid_start, grid_end, step_num):
            # comput the SASV_eer on the calibrated data
            fused_scores = fuse_llr_nonlinear(data[:, 0], data[:, 1], rho)
            eer_tmp, _ = get_metric(fused_scores, labels)

            # if it is the best
            if eer_tmp <= eer:
                eer = eer_tmp
                rho_best = rho
    
        # update the search range
        if (grid_end - rho_best) < (rho_best - grid_start):
            grid_start = rho_best
            if grid_end == rho_best:
                # no need to search further
                break
        else:
            grid_end = rho_best
            if grid_start == rho_best:
                break

    return rho_best





if __name__ == "__main__":

    label_dic = {'tar': args.label_value_target,
                 'non': args.label_value_nontarget,
                 'spf': args.label_value_spoof}

    ###
    # Load and compose data
    ###
    # dev set
    score_label_dev = pd.read_csv(args.dev_input_csv)
    # evaluation data
    score_label_eval = pd.read_csv(args.eval_input_csv)
    
    # stack the scores into 2d vectors [i, 0] --> ASV, [i, 1] --> CM
    X_dev = score_label_dev[['asv_score', 'cm_score']].to_numpy()
    X_dev_label = score_label_dev['sasv_label'].to_numpy()

    # we further split the dev data into the three classes.
    # They will be used to fit three Gaussians
    X_tar_dev = score_label_dev.query(
        "sasv_label == {:f}".format(label_dic['tar']))[['asv_score', 'cm_score']].to_numpy()
    X_non_dev = score_label_dev.query(
        "sasv_label == {:f}".format(label_dic['non']))[['asv_score', 'cm_score']].to_numpy()
    X_spf_dev = score_label_dev.query(
        "sasv_label == {:f}".format(label_dic['spf']))[['asv_score', 'cm_score']].to_numpy()

    # eval set
    X_eval = score_label_eval[['asv_score', 'cm_score']].to_numpy()
    
    if not args.use_linear_fusion:
        print("Fusion using a non-linear function")
    else:
        print("Fusion using a linear function")
    print("Development data size: ", X_dev.shape)
    print("Evaluation data size: ", X_eval.shape)

    ## 
    # fit gaussians and get LLRs
    ##
    # target bonafide data
    gm_tar = fit_gaussians(X_tar_dev)
    gm_non = fit_gaussians(X_non_dev)
    gm_spf = fit_gaussians(X_spf_dev)

    # 
    # compute the LLRs for dev data
    X_dev_llrs = get_llrs(X_dev, gm_tar, gm_non, gm_spf)

    ##
    # get calibration functions based on logistic reg
    ##
    # for calibrating CM LLRs
    # target and nontarget are positive class data, spoofed is negative
    cm_label = X_dev_label != label_dic['spf']    

    # X_dev_llrs[:, 1] is the CM LLRs
    cm_data = X_dev_llrs[:, 1]

    # get the calibration function
    func_cm_calib = get_calibration_affine(cm_data, cm_label)

    # for calibrating ASV LLRs
    # target data is positive, nontarget data is negative
    #  we ignore spoofed data here
    asv_label = X_dev_label[cm_label] != label_dic['non']

    # X_dev_llrs[:, 0] is the ASV LLRs
    asv_data = X_dev_llrs[cm_label, 0]

    # get the calibration function
    func_asv_calib = get_calibration_affine(asv_data, asv_label)
    
    # get calibrated data on dev set
    X_dev_calib_llrs = np.zeros_like(X_dev_llrs)
    #  calibrate LLR for asv
    X_dev_calib_llrs[:, 0] = func_asv_calib(X_dev_llrs[:, 0])
    #  calibrate LLR for cm
    X_dev_calib_llrs[:, 1] = func_cm_calib(X_dev_llrs[:, 1])

    
    ##
    # search for the best rho parameter on development set
    ##
    if not args.use_linear_fusion:
        # only non-linear fusion need rho
        rho = grid_search_rho(X_dev_calib_llrs, X_dev_label)
    else:
        rho = None


    ##
    # get calibrated LLRs on evaluation data
    ##
    # compute the LLRs for evaluation data
    X_eval_llrs = get_llrs(X_eval, gm_tar, gm_non, gm_spf)
    
    # get calibrated data on dev set
    X_eval_calib_llrs = np.zeros_like(X_eval_llrs)
    #  calibrate LLR for asv
    X_eval_calib_llrs[:, 0] = func_asv_calib(X_eval_llrs[:, 0])
    #  calibrate LLR for cm
    X_eval_calib_llrs[:, 1] = func_cm_calib(X_eval_llrs[:, 1])

    ##
    # fusion
    ##
    if args.use_linear_fusion:
        X_eval_sasv_scores = fuse_llr_linear(
            X_eval_calib_llrs[:, 0], X_eval_calib_llrs[:, 1])
    else:
        X_eval_sasv_scores = fuse_llr_nonlinear(
            X_eval_calib_llrs[:, 0], X_eval_calib_llrs[:, 1], rho)

    score_label_eval['fused_score'] = pd.Series(X_eval_sasv_scores)

    
    if score_label_eval['fused_score'].isnull().any().any():
        print("Fused scores contain invalid value. Try linear fusion")
    else:
        print("Save fused scores to ", args.eval_output_csv)
        score_label_eval.to_csv(args.eval_output_csv, index=False)
