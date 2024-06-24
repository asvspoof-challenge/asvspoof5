import os
import numpy as np

from calculate_modules import *
import a_dcf


def calculate_minDCF_EER_CLLR(cm_scores, cm_keys, output_file, printout=True):
    """
    Evaluation metrics for track 1
    Primary metrics: min DCF,
    Secondary metrics: EER, CLLR
    """
    
    Pspoof = 0.05
    dcf_cost_model = {
        'Pspoof': Pspoof,  # Prior probability of a spoofing attack
        'Cmiss': 1,  # Cost of CM system falsely rejecting target speaker
        'Cfa' : 10, # Cost of CM system falsely accepting nontarget speaker
    }


    assert cm_keys.size == cm_scores.size, "Error, unequal length of cm label and score files"
    
    # Extract bona fide (real human) and spoof scores from the CM scores
    bona_cm = cm_scores[cm_keys == 'bonafide']
    spoof_cm = cm_scores[cm_keys == 'spoof']

    # EERs of the standalone systems
    eer_cm, frr, far, thresholds, eer_threshold = compute_eer(bona_cm, spoof_cm)#[0]
    # cllr
    cllr_cm = calculate_CLLR(bona_cm, spoof_cm)
    # min DCF
    minDCF_cm, _ = compute_mindcf(frr, far, thresholds, Pspoof, dcf_cost_model['Cmiss'], dcf_cost_model['Cfa'])
    # actual DCF
    actDCF, _ = compute_actDCF(bona_cm, spoof_cm, Pspoof, dcf_cost_model['Cmiss'], dcf_cost_model['Cfa'])

    if printout:
        with open(output_file, "w") as f_res:
            f_res.write('\nCM SYSTEM\n')
            f_res.write('\tmin DCF \t\t= {} '
                        '(min DCF for countermeasure)\n'.format(
                            minDCF_cm))
            f_res.write('\tEER\t\t= {:8.9f} % '
                        '(EER for countermeasure)\n'.format(
                            eer_cm * 100))
            f_res.write('\tCLLR\t\t= {:8.9f} bits '
                        '(CLLR for countermeasure)\n'.format(
                            cllr_cm))
            f_res.write('\tactDCF\t\t= {:} '
                        '(actual DCF)\n'.format(
                            actDCF))
        os.system(f"cat {output_file}")

    return minDCF_cm, eer_cm, cllr_cm, actDCF


def calculate_aDCF_tdcf_tEER(cm_scores, asv_scores, sasv_scores, cm_keys,
                             asv_keys, output_file, printout=True):
    """
    Evaluation metrics for Phase 2
    Primary metrics: a_DCF
    Secondary metrics: min t-DCF, t-EER
    """

    ###
    # configurations
    ###
    # for ASV-constrained tDCF
    Pspoof = 0.05
    tdcf_cost_model = {
        # Prior probability of a spoofing attack
        'Pspoof': Pspoof,
        # Prior probability of target speaker
        'Ptar': (1 - Pspoof) * 0.99,
        # Prior probability of nontarget speaker
        'Pnon': (1 - Pspoof) * 0.01,
        # Cost of CM falsely rejecting bonafide
        #  or tandem system rejecting target bonafide
        'Cmiss': 1,
        # Cost of CM system falsely accepting spoofed
        #  or tandem system falsely accepting nontarget bonafide
        'Cfa': 10,
        # Cost of tandem system falsely accepting spoofed
        'Cfa_spoof': 10,
        # 
        # legacy tDCF Cost of ASV system falsely rejecting target speaker
        'Cmiss_asv': 1,
        # legacy tDCF Cost of ASV system falsely accepting nontarget speaker
        'Cfa_asv': 10,
        # legacy tDCF Cost of CM system falsely rejecting target speaker
        'Cmiss_cm': 1,
        # legacy tDCF # Cost of CM system falsely accepting spoof
        'Cfa_cm': 10,  
    }

    # from default ASV of organizers
    Pfa_non_ASV_org =  0.01881016557566423
    Pmiss_ASV_org = 0.01880141010575793
    Pfa_spf_ASV_org = 0.4607082907604729

    # for a-DCF
    cost_adcf = a_dcf.CostModel(
        Ptrg = tdcf_cost_model['Ptar'],
        Pnontrg = tdcf_cost_model['Pnon'],
        Pspf = tdcf_cost_model['Pspoof'],
        Cmiss = tdcf_cost_model['Cmiss'],
        Cfa_asv = tdcf_cost_model['Cfa'],
        Cfa_cm = tdcf_cost_model['Cfa_spoof'])

    ###
    # load scores
    ###
    
    # Extract SASV scores
    tar_sasv = sasv_scores[asv_keys == 'target']
    nontar_sasv = sasv_scores[asv_keys == 'nontarget']
    spoof_sasv = sasv_scores[asv_keys == 'spoof']
    
    ###
    # Calculate a-DCF
    ###
    # input should be SASV scores
    
    adcf = a_dcf._calculate_a_dcf(
        tar_sasv, nontar_sasv, spoof_sasv, cost_adcf, False)['min_a_dcf']


    if asv_scores is None or cm_scores is None:
        # no scores provided for ASV or CM, quit
        if printout:
            with open(output_file, "w") as f_res:
                f_res.write('\nSASV RESULT\n')
                f_res.write('\ta-DCF\t\t= {:8.9f}\n'.format(adcf))
            os.system(f"cat {output_file}")
        return adcf

    ###
    # load scores for secondary metrics
    ###
    
    # Extract target, nontarget, and spoof scores from the ASV scores
    tar_asv = asv_scores[asv_keys == 'target']
    non_asv = asv_scores[asv_keys == 'nontarget']
    spoof_asv = asv_scores[asv_keys == 'spoof']

    # Extract bona fide (real human) and spoof scores from the CM scores
    bona_cm = cm_scores[cm_keys == 'bonafide']
    spoof_cm = cm_scores[cm_keys == 'spoof']
    
    # need to check here that the utt_ids are the same in cm and asv files
    # --> this is checked in util
    # asv-cm score pairs
    X_tar = np.array([asv_scores[asv_keys == 'target'],
                      cm_scores[asv_keys == 'target']], dtype=object)
    X_non = np.array([asv_scores[asv_keys == 'nontarget'],
                      cm_scores[asv_keys == 'nontarget']], dtype=object)
    X_spf = np.array([asv_scores[asv_keys == 'spoof'],
                      cm_scores[asv_keys == 'spoof']], dtype=object)
    ###
    # Fix tandem detection cost function (t-DCF) parameters
    ###

    # Compute t-DCF, given default ASV error rates
    tDCF_curve, _ = compute_tDCF(
        bona_cm, spoof_cm, Pfa_non_ASV_org, Pmiss_ASV_org, Pfa_spf_ASV_org,
        tdcf_cost_model, print_cost=False)
    
    # Minimum t-DCF
    min_tDCF_index = np.argmin(tDCF_curve)
    min_tDCF = tDCF_curve[min_tDCF_index]

    ###
    # t-EER
    ###

    # Obtain ASV error curves and ASV thresholds
    Pmiss_ASV, Pfa_non_ASV, Pfa_spf_ASV, tau_ASV = compute_Pmiss_Pfa_Pspoof_curves(X_tar[0], X_non[0], X_spf[0])

    # Obtain CM error curves and CM thresholds.
    Pmiss_CM, Pfa_CM, tau_CM = compute_det_curve(np.concatenate([X_tar[1], X_non[1]]), X_spf[1])

    teer = compute_teer(Pmiss_CM, Pfa_CM, tau_CM, Pmiss_ASV, Pfa_non_ASV, Pfa_spf_ASV, tau_ASV)
    

    if printout:
        with open(output_file, "w") as f_res:
            f_res.write('\nSASV RESULT\n')
            f_res.write('\ta-DCF\t\t= {:8.9f}\n'.format(adcf))
            f_res.write('\tt-EER\t\t= {:8.9f}\n'.format(teer))
            f_res.write('\tmin-tDCF\t\t= {:8.9f}\n'.format(min_tDCF))
        os.system(f"cat {output_file}")

    return adcf, min_tDCF, teer


