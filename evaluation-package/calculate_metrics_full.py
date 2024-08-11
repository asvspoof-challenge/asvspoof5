#!/usr/bin/env python3
"""
main function to run evaluation package using
the full meta key table

"""
import sys
import argparse
import numpy as np

import util
from calculate_metrics import calculate_minDCF_EER_CLLR_actDCF
from calculate_metrics import calculate_aDCF_tdcf_tEER


def calculate_minDCF_EER_CLLR_actDCF_full(cm_score_key_pd):
    """
    compute metrics for track 1, loop over all sub conditions

    input
    -----
      cm_score_key_pd: dataframe, with score and full set of meta data

    output
    ------
      attack_list: list of str, list of attack labels
      codec_list: list of str, list of codec labels

      minDCF_mat: np.array, in shape [len(attack_list), len(codec_list)]
      eer_mat: np.array, in shape [len(attack_list), len(codec_list)]
      cllr_mat: np.array, in shape [len(attack_list), len(codec_list)]
      actDCF_mat: np.array, in shape [len(attack_list), len(codec_list)]
    """
    # return lists of attacks and codes and add pooled case
    attack_list = util.return_attacks(cm_score_key_pd)
    attack_list.insert(0, util.g_pooled_label)
    
    # return lists of attacks and codes and add pooled case
    codec_list = util.return_codecs(cm_score_key_pd)
    codec_list.insert(0, util.g_pooled_label)

    # prepare the output matrics
    minDCF_mat = np.zeros([len(attack_list), len(codec_list)]) * np.nan
    eer_mat = np.zeros_like(minDCF_mat) * np.nan
    cllr_mat = np.zeros_like(minDCF_mat) * np.nan
    actDCF_mat = np.zeros_like(minDCF_mat) * np.nan

    
    # for each codec
    for idx2, codec in enumerate(codec_list):
        if codec == util.g_pooled_label:
            # pooled over codec
            pd_tmp = cm_score_key_pd
        else:
            # retrieve data from a specific codec
            query = '{:s} == "{:s}"'.format(util.g_codec_tag, codec)
            pd_tmp = cm_score_key_pd.query(query)
            
        # get bona fide score and labels
        query = '{:s} == "{:s}"'.format(util.g_cm_label_tag, util.g_cm_bon)
        # to be compatible with calculate_metrics.py functions,
        # we create the score and label arrays
        bon_scores = pd_tmp.query(query)[util.g_cm_score_tag].to_numpy()
        bon_labels = np.array([util.g_cm_bon for x in bon_scores])

        # for each spoofing attack
        for idx1, attack in enumerate(attack_list):
            
            if attack == util.g_pooled_label:
                # pooled over attacks
                query = '{:s} == "{:s}"'.format(util.g_cm_label_tag, util.g_cm_spf)
            else:
                # a specific attack
                query = '{:s} == "{:s}"'.format(util.g_attack_tag, attack)
                
            spf_scores = pd_tmp.query(query)[util.g_cm_score_tag].to_numpy()
            spf_labels = np.array([util.g_cm_spf for x in spf_scores])

            # compose
            # to be compatible with calculate_metrics.py functions,
            # we concatenate the score and label tuple
            cm_scores = np.concatenate([bon_scores, spf_scores], axis=0)
            cm_keys = np.concatenate([bon_labels, spf_labels], axis=0)

            # compute metrics
            if len(bon_scores) and len(spf_scores):
                
                minDCF, eer, cllr, actDCF = calculate_minDCF_EER_CLLR_actDCF(
                    cm_scores = cm_scores,
                    cm_keys = cm_keys,
                    output_file='', printout=False)
                
                # save
                minDCF_mat[idx1, idx2] = minDCF
                eer_mat[idx1, idx2] = eer
                cllr_mat[idx1, idx2] = cllr
                actDCF_mat[idx1, idx2] = actDCF

    return attack_list, codec_list, minDCF_mat, eer_mat, cllr_mat, actDCF_mat


def calculate_aDCF_tdcf_tEER_full(sasv_score_key_pd, asv_org_score_key_pd, flag_sasv_only=False):
    """
    compute metrics for track 2, loop over all sub conditions

    input
    -----
      sasv_score_key_pd: dataframe, with score and full set of meta data
      asv_org_score_key_pd: dataframe, with score from ASV organizers
      flag_sasv_only: bool, whether assume input contains SASV score only

    output
    ------
      attack_list: list of str, list of attack labels
      codec_list: list of str, list of codec labels

      adcf_mat: np.array, in shape [len(attack_list), len(codec_list)]
      min_tDCF_mat: np.array, in shape [len(attack_list), len(codec_list)]
      teer_mat: np.array, in shape [len(attack_list), len(codec_list)]
    """
    
    # return lists of attacks and codes and add pooled case
    attack_list = util.return_attacks(sasv_score_key_pd)
    attack_list.insert(0, util.g_pooled_label)
    
    # return lists of attacks and codes and add pooled case
    codec_list = util.return_codecs(sasv_score_key_pd)
    codec_list.insert(0, util.g_pooled_label)

    # prepare the output matrics
    adcf_mat = np.zeros([len(attack_list), len(codec_list)]) * np.nan
    min_tDCF_mat = np.zeros_like(adcf_mat) * np.nan
    teer_mat = np.zeros_like(adcf_mat) * np.nan

    
    # for each codec
    for idx2, codec in enumerate(codec_list):
        print('\n' + codec, end=' ', flush=True)
        if codec == util.g_pooled_label:
            # pooled over codec
            pd_tmp = sasv_score_key_pd
            if asv_org_score_key_pd is not None:
                asv_org_pd_tmp = asv_org_score_key_pd
        else:
            # retrieve data from a specific codec
            query = '{:s} == "{:s}"'.format(util.g_codec_tag, codec)
            pd_tmp = sasv_score_key_pd.query(query)
            if asv_org_score_key_pd is not None:
                asv_org_pd_tmp = asv_org_score_key_pd.query(query)
            
        # bona target 
        query = '{:s} == "{:s}"'.format(util.g_asv_label_tag, util.g_asv_tar)
        bon_pd = pd_tmp.query(query)
        if asv_org_score_key_pd is not None:
            asv_org_bon_pd = asv_org_pd_tmp.query(query)

        # bon nontaret
        query = '{:s} == "{:s}"'.format(util.g_asv_label_tag, util.g_asv_non)
        non_pd = pd_tmp.query(query)
        if asv_org_score_key_pd is not None:
            asv_org_non_pd = asv_org_pd_tmp.query(query)
        
        # for each spoofing attack
        for idx1, attack in enumerate(attack_list):
            print(attack, end=' ', flush=True)
            if attack == util.g_pooled_label:
                # pooled over all attacks
                query = '{:s} == "{:s}"'.format(util.g_cm_label_tag, util.g_cm_spf)
            else:
                # a single attack
                query = '{:s} == "{:s}"'.format(util.g_attack_tag, attack)

            # spoofed
            spf_pd = pd_tmp.query(query)
            if asv_org_score_key_pd is not None:
                asv_org_spf_pd = asv_org_pd_tmp.query(query)
            
            if bon_pd.shape[0] and non_pd.shape[0] and spf_pd.shape[0]:

                # compose score and label vectors
                if flag_sasv_only:
                    cm_scores = None
                    cm_keys = None
                    asv_scores = None
                else:
                    cm_scores = np.concatenate([
                        bon_pd[util.g_cm_score_tag].to_numpy(),
                        non_pd[util.g_cm_score_tag].to_numpy(),
                        spf_pd[util.g_cm_score_tag].to_numpy()], axis=0)
                    cm_keys = np.concatenate([
                        bon_pd[util.g_cm_label_tag].to_numpy(),
                        non_pd[util.g_cm_label_tag].to_numpy(),
                        spf_pd[util.g_cm_label_tag].to_numpy()], axis=0)
                    asv_scores = np.concatenate([
                        bon_pd[util.g_asv_score_tag].to_numpy(),
                        non_pd[util.g_asv_score_tag].to_numpy(),
                        spf_pd[util.g_asv_score_tag].to_numpy()], axis=0)
                
                asv_keys = np.concatenate([
                    bon_pd[util.g_asv_label_tag].to_numpy(),
                    non_pd[util.g_asv_label_tag].to_numpy(),
                    spf_pd[util.g_asv_label_tag].to_numpy()], axis=0)
                sasv_scores = np.concatenate([
                    bon_pd[util.g_sasv_score_tag].to_numpy(),
                    non_pd[util.g_sasv_score_tag].to_numpy(),
                    spf_pd[util.g_sasv_score_tag].to_numpy()], axis=0)
                
                # asv scores from the organizer
                if asv_org_score_key_pd is not None and not flag_sasv_only:
                    asv_scores_org = np.concatenate([
                        asv_org_bon_pd[util.g_asv_score_tag].to_numpy(),
                        asv_org_non_pd[util.g_asv_score_tag].to_numpy(),
                        asv_org_spf_pd[util.g_asv_score_tag].to_numpy()], axis=0)
                    asv_keys_org = np.concatenate([
                        asv_org_bon_pd[util.g_asv_label_tag].to_numpy(),
                        asv_org_non_pd[util.g_asv_label_tag].to_numpy(),
                        asv_org_spf_pd[util.g_asv_label_tag].to_numpy()], axis=0)
                else:
                    asv_scores_org = None
                    asv_keys_org = None

                
                results = calculate_aDCF_tdcf_tEER(
                    cm_scores = cm_scores,
                    cm_keys = cm_keys,
                    asv_scores = asv_scores,
                    asv_keys = asv_keys,
                    sasv_scores = sasv_scores,
                    asv_scores_org = asv_scores_org,
                    asv_keys_org = asv_keys_org,
                    output_file='', printout=False)
                
                # save
                if flag_sasv_only:
                    adcf_mat[idx1, idx2] = results
                else:
                    adcf_mat[idx1, idx2] = results[0]
                    min_tDCF_mat[idx1, idx2] = results[1]
                    teer_mat[idx1, idx2] = results[2]

    return attack_list, codec_list, adcf_mat, min_tDCF_mat, teer_mat
