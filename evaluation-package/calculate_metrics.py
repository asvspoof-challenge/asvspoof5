import sys
import os

import numpy as np

from a_dcf import a_dcf
from .calculate_modules import *


def calculate_minDCF_EER_CLLR(cm_scores_file,
                       output_file,
                       printout=True):
    # Evaluation metrics for Phase 1
    # Primary metrics: min DCF,
    # Secondary metrics: EER, CLLR

    Pspoof = 0.05
    dcf_cost_model = {
        'Pspoof': Pspoof,  # Prior probability of a spoofing attack
        'Cmiss': 1,  # Cost of CM system falsely rejecting target speaker
        'Cfa' : 10, # Cost of CM system falsely accepting nontarget speaker
    }


    # Load CM scores
    cm_data = np.genfromtxt(cm_scores_file, dtype=str)
    cm_sources = cm_data[:, 2]
    cm_keys = cm_data[:, 3]
    cm_scores = cm_data[:, 4].astype(np.float64)

    # Extract bona fide (real human) and spoof scores from the CM scores
    bona_cm = cm_scores[cm_keys == 'bonafide']
    spoof_cm = cm_scores[cm_keys == 'spoof']

    # EERs of the standalone systems and fix ASV operating point to EER threshold
    eer_cm, frr, far, thresholds = compute_eer(bona_cm, spoof_cm)[0]
    cllr_cm = calculate_CLLR(bona_cm, spoof_cm)
    minDCF_cm, _ = compute_mindcf(frr, far, thresholds, Pspoof, dcf_cost_model['Cmiss'], dcf_cost_model['Cfa'])

    attack_types = [f'A{_id:02d}' for _id in range(9, 17)]
    if printout:
        spoof_cm_breakdown = {
            attack_type: cm_scores[cm_sources == attack_type]
            for attack_type in attack_types
        }

        eer_cm_breakdown = {
            attack_type: compute_eer(bona_cm,
                                     spoof_cm_breakdown[attack_type])[0]
            for attack_type in attack_types
        }

    if printout:
        with open(output_file, "w") as f_res:
            f_res.write('\nCM SYSTEM\n')
            f_res.write('\tmin DCF \t\t= {} % '
                        '(min DCF for countermeasure)\n'.format(
                            minDCF_cm))
            f_res.write('\tEER\t\t= {:8.9f} % '
                        '(EER for countermeasure)\n'.format(
                            eer_cm * 100))
            f_res.write('\tCLLR\t\t= {:8.9f} % '
                        '(CLLR for countermeasure)\n'.format(
                            cllr_cm * 100))

            f_res.write('\nBREAKDOWN CM SYSTEM\n')
            for attack_type in attack_types:
                _eer = eer_cm_breakdown[attack_type] * 100
                f_res.write(
                    f'\tEER {attack_type}\t\t= {_eer:8.9f} % (Equal error rate for {attack_type})\n'
                )
        os.system(f"cat {output_file}")

    return minDCF_cm, eer_cm, cllr_cm


def calculate_aDCF_tdcf_tEER(cm_scores_file,
                       asv_score_file,
                       output_file,
                       printout=True):
    # Evaluation metrics for Phase 2
    # Primary metrics: a_DCF
    # Secondary metrics: min t-DCF, t-EER

    # Calculate a-DCF (only one score file, the output of the integrated/tandem system is needed)
    adcf = a_dcf.calculate_a_dcf(cm_scores_file)['min_a_dcf']

    # Fix tandem detection cost function (t-DCF) parameters
    Pspoof = 0.05
    tdcf_cost_model = {
        'Pspoof': Pspoof,  # Prior probability of a spoofing attack
        'Ptar': (1 - Pspoof) * 0.99,  # Prior probability of target speaker
        'Pnon': (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker
        'Cmiss': 1,  # Cost of ASV system falsely rejecting target speaker
        'Cfa': 10,  # Cost of ASV system falsely accepting nontarget speaker
        'Cmiss_asv': 1,  # Cost of ASV system falsely rejecting target speaker
        'Cfa_asv': 10,  # Cost of ASV system falsely accepting nontarget speaker
        'Cmiss_cm': 1,  # Cost of CM system falsely rejecting target speaker
        'Cfa_cm': 10,  # Cost of CM system falsely accepting spoof
    }

    # Load organizers' ASV scores
    asv_data = np.genfromtxt(asv_score_file, dtype=str)
    asv_keys = asv_data[:, 1]
    asv_scores = asv_data[:, 2].astype(np.float64)

    # Load CM scores
    cm_data = np.genfromtxt(cm_scores_file, dtype=str)
    cm_sources = cm_data[:, 2]
    cm_keys = cm_data[:, 3]
    cm_scores = cm_data[:, 4].astype(np.float64)

    # Extract target, nontarget, and spoof scores from the ASV scores
    tar_asv = asv_scores[asv_keys == 'target']
    non_asv = asv_scores[asv_keys == 'nontarget']
    spoof_asv = asv_scores[asv_keys == 'spoof']

    # Extract bona fide (real human) and spoof scores from the CM scores
    bona_cm = cm_scores[cm_keys == 'bonafide']
    spoof_cm = cm_scores[cm_keys == 'spoof']
   

    # need to check here that the utt_ids are the same in cm and asv files
    X_tar = np.array([asv_scores[asv_keys == 'target'], cm_scores[asv_keys == 'target']],dtype=object)
    X_non = np.array([asv_scores[asv_keys == 'nontarget'], cm_scores[asv_keys == 'nontarget']],dtype=object)
    X_spf = np.array([asv_scores[asv_keys == 'spoof'], cm_scores[asv_keys == 'spoof']],dtype=object)

    # Obtain ASV error curves and ASV thresholds
    Pmiss_ASV, Pfa_non_ASV, Pfa_spf_ASV, tau_ASV = compute_Pmiss_Pfa_Pspoof_curves(X_tar[0], X_non[0], X_spf[0])

    # Obtain CM error curves and CM thresholds.
    Pmiss_CM, Pfa_CM, tau_CM = compute_det_curve(np.concatenate([X_tar[1], X_non[1]]), X_spf[1])

    # EERs of the standalone systems and fix ASV operating point to
    # EER threshold
    eer_asv, _, _, asv_threshold = compute_eer(tar_asv, non_asv)
    eer_cm, frr, far, thresholds = compute_eer(bona_cm, spoof_cm)[0]
    minDCF_cm, _ = compute_mindcf(frr, far, thresholds, Pspoof, tdcf_cost_model['Cmiss'], tdcf_cost_model['Cfa'])


    #attack_types = [f'A{_id:02d}' for _id in range(7, 20)]
    attack_types = [f'A{_id:02d}' for _id in range(9, 17)]
    if printout:
        spoof_cm_breakdown = {
            attack_type: cm_scores[cm_sources == attack_type]
            for attack_type in attack_types
        }

        eer_cm_breakdown = {
            attack_type: compute_eer(bona_cm,
                                     spoof_cm_breakdown[attack_type])[0]
            for attack_type in attack_types
        }

    [Pfa_asv, Pmiss_asv,
     Pmiss_spoof_asv, Pfa_spoof_asv] = obtain_asv_error_rates(tar_asv, non_asv, spoof_asv,
                                               asv_threshold)

    # Compute t-DCF
    tDCF_curve, _ = compute_tDCF(bona_cm,
                                             spoof_cm,
                                             Pfa_asv,
                                             Pmiss_asv,
                                             Pmiss_spoof_asv,
                                             tdcf_cost_model,
                                             print_cost=False)

    # Minimum t-DCF
    min_tDCF_index = np.argmin(tDCF_curve)
    min_tDCF = tDCF_curve[min_tDCF_index]

    if printout:
        with open(output_file, "w") as f_res:
            f_res.write('\nSASV RESULT\n')
            f_res.write('\tEER\t\t= {:8.9f} % '
                        '(Equal error rate for countermeasure)\n'.format(
                            eer_cm * 100))

            f_res.write('\nTANDEM\n')
            f_res.write('\tmin-tDCF\t\t= {:8.9f}\n'.format(min_tDCF))

            f_res.write('\nBREAKDOWN CM SYSTEM\n')
            for attack_type in attack_types:
                _eer = eer_cm_breakdown[attack_type] * 100
                f_res.write(
                    f'\tEER {attack_type}\t\t= {_eer:8.9f} % (Equal error rate for {attack_type})\n'
                )
        os.system(f"cat {output_file}")

    teer = compute_teer(Pmiss_CM, Pfa_CM, tau_CM, Pmiss_asv, Pfa_non_ASV, Pfa_spf_ASV, tau_ASV)
    
    return adcf, min_tDCF, teer


