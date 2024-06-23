import sys
import numpy as np


def obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold):

    # False alarm and miss rates for ASV
    Pfa_asv = sum(non_asv >= asv_threshold) / non_asv.size
    Pmiss_asv = sum(tar_asv < asv_threshold) / tar_asv.size

    # Rate of rejecting spoofs in ASV
    if spoof_asv.size == 0:
        Pmiss_spoof_asv = None
        Pfa_spoof_asv = None
    else:
        Pmiss_spoof_asv = np.sum(spoof_asv < asv_threshold) / spoof_asv.size
        Pfa_spoof_asv = np.sum(spoof_asv >= asv_threshold) / spoof_asv.size

    return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv


def obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold):

    # False alarm and miss rates for ASV
    Pfa_asv = sum(non_asv >= asv_threshold) / non_asv.size
    Pmiss_asv = sum(tar_asv < asv_threshold) / tar_asv.size

    # Rate of rejecting spoofs in ASV
    if spoof_asv.size == 0:
        Pmiss_spoof_asv = None
        Pfa_spoof_asv = None
    else:
        Pmiss_spoof_asv = np.sum(spoof_asv < asv_threshold) / spoof_asv.size
        Pfa_spoof_asv = np.sum(spoof_asv >= asv_threshold) / spoof_asv.size

    return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv


def compute_det_curve(target_scores, nontarget_scores):

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - \
        (np.arange(1, n_scores + 1) - tar_trial_sums)

    # false rejection rates
    frr = np.concatenate(
        (np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums /
                          nontarget_scores.size))  # false acceptance rates
    # Thresholds are the sorted scores
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))

    return frr, far, thresholds


def compute_Pmiss_Pfa_Pspoof_curves(tar_scores, non_scores, spf_scores):

    # Concatenate all scores and designate arbitrary labels 1=target, 0=nontarget, -1=spoof
    all_scores = np.concatenate((tar_scores, non_scores, spf_scores))
    labels = np.concatenate((np.ones(tar_scores.size), np.zeros(non_scores.size), -1*np.ones(spf_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Cumulative sums
    tar_sums    = np.cumsum(labels==1)
    non_sums    = np.cumsum(labels==0)
    spoof_sums  = np.cumsum(labels==-1)

    Pmiss       = np.concatenate((np.atleast_1d(0), tar_sums / tar_scores.size))
    Pfa_non     = np.concatenate((np.atleast_1d(1), 1 - (non_sums / non_scores.size)))
    Pfa_spoof   = np.concatenate((np.atleast_1d(1), 1 - (spoof_sums / spf_scores.size)))
    thresholds  = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores

    return Pmiss, Pfa_non, Pfa_spoof, thresholds


def compute_eer(target_scores, nontarget_scores):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, frr, far, thresholds, thresholds[min_index]


def compute_mindcf(frr, far, thresholds, Pspoof, Cmiss, Cfa):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds

    p_target = 1- Pspoof
    for i in range(0, len(frr)):
        # Weighted sum of false negative and false positive errors.
        c_det = Cmiss * frr[i] * p_target + Cfa * far[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(Cmiss * p_target, Cfa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold

def compute_actDCF(bonafide_scores, spoof_scores, Pspoof, Cmiss, Cfa):
    """
    compute actual DCF, given threshold decided by prior and decision costs

    input
    -----
      bonafide_scores: np.array, scores of bonafide data
      spoof_scores: np.array, scores of spoof data
      Pspoof: scalar, prior probabiltiy of spoofed class
      Cmiss: scalar, decision cost of missing a bonafide sample
      Cfa: scalar, decision cost of falsely accept a spoofed sample

    output
    ------
      actDCF: scalar, actual DCF normalized
      threshold: scalar, threshold for making the decision
    """
    # the beta in evaluation plan (eq.(3))
    beta = Cmiss * (1 - Pspoof) / (Cfa * Pspoof)
    
    # compute the decision threshold based on
    threshold = - np.log(beta)

    # miss rate
    rate_miss = np.sum(bonafide_scores < threshold) / bonafide_scores.size

    # fa rate
    rate_fa = np.sum(spoof_scores >= threshold) / spoof_scores.size

    # unnormalized DCF
    act_dcf = Cmiss * (1 - Pspoof) * rate_miss + Cfa * Pspoof * rate_fa

    # normalized DCF
    act_dcf = act_dcf / np.min([Cfa * Pspoof, Cmiss * (1 - Pspoof)])
    
    return act_dcf, threshold
    


def compute_tDCF(bonafide_score_cm, spoof_score_cm, Pfa_asv, Pmiss_asv,
                 Pmiss_spoof_asv, cost_model, print_cost):

    # Sanity check of cost parameters
    if cost_model['Cfa_asv'] < 0 or cost_model['Cmiss_asv'] < 0 or \
            cost_model['Cfa_cm'] < 0 or cost_model['Cmiss_cm'] < 0:
        print('WARNING: Usually the cost values should be positive!')

    if cost_model['Ptar'] < 0 or cost_model['Pnon'] < 0 or cost_model['Pspoof'] < 0 or \
            np.abs(cost_model['Ptar'] + cost_model['Pnon'] + cost_model['Pspoof'] - 1) > 1e-10:
        sys.exit(
            'ERROR: Your prior probabilities should be positive and sum up to one.'
        )

    # Unless we evaluate worst-case model, we need to have some spoof tests against asv
    if Pmiss_spoof_asv is None:
        sys.exit(
            'ERROR: you should provide miss rate of spoof tests against your ASV system.'
        )

    # Sanity check of scores
    combined_scores = np.concatenate((bonafide_score_cm, spoof_score_cm))
    if np.isnan(combined_scores).any() or np.isinf(combined_scores).any():
        sys.exit('ERROR: Your scores contain nan or inf.')

    # Sanity check that inputs are scores and not decisions
    n_uniq = np.unique(combined_scores).size
    if n_uniq < 3:
        sys.exit(
            'ERROR: You should provide soft CM scores - not binary decisions')

    # Obtain miss and false alarm rates of CM
    Pmiss_cm, Pfa_cm, CM_thresholds = compute_det_curve(
        bonafide_score_cm, spoof_score_cm)

    # Constants - see ASVspoof 2019 evaluation plan
    C1 = cost_model['Ptar'] * (cost_model['Cmiss_cm'] - cost_model['Cmiss_asv'] * Pmiss_asv) - \
        cost_model['Pnon'] * cost_model['Cfa_asv'] * Pfa_asv
    C2 = cost_model['Cfa_cm'] * cost_model['Pspoof'] * (1 - Pmiss_spoof_asv)

    # Sanity check of the weights
    if C1 < 0 or C2 < 0:
        sys.exit(
            'You should never see this error but I cannot evalute tDCF with negative weights - please check whether your ASV error rates are correctly computed?'
        )

    # Obtain t-DCF curve for all thresholds
    tDCF = C1 * Pmiss_cm + C2 * Pfa_cm

    # Normalized t-DCF
    tDCF_norm = tDCF / np.minimum(C1, C2)

    # Everything should be fine if reaching here.
    if print_cost:

        print('t-DCF evaluation from [Nbona={}, Nspoof={}] trials\n'.format(
            bonafide_score_cm.size, spoof_score_cm.size))
        print('t-DCF MODEL')
        print('   Ptar         = {:8.5f} (Prior probability of target user)'.
              format(cost_model['Ptar']))
        print(
            '   Pnon         = {:8.5f} (Prior probability of nontarget user)'.
            format(cost_model['Pnon']))
        print(
            '   Pspoof       = {:8.5f} (Prior probability of spoofing attack)'.
            format(cost_model['Pspoof']))
        print(
            '   Cfa_asv      = {:8.5f} (Cost of ASV falsely accepting a nontarget)'
            .format(cost_model['Cfa_asv']))
        print(
            '   Cmiss_asv    = {:8.5f} (Cost of ASV falsely rejecting target speaker)'
            .format(cost_model['Cmiss_asv']))
        print(
            '   Cfa_cm       = {:8.5f} (Cost of CM falsely passing a spoof to ASV system)'
            .format(cost_model['Cfa_cm']))
        print(
            '   Cmiss_cm     = {:8.5f} (Cost of CM falsely blocking target utterance which never reaches ASV)'
            .format(cost_model['Cmiss_cm']))
        print(
            '\n   Implied normalized t-DCF function (depends on t-DCF parameters and ASV errors), s=CM threshold)'
        )

        if C2 == np.minimum(C1, C2):
            print(
                '   tDCF_norm(s) = {:8.5f} x Pmiss_cm(s) + Pfa_cm(s)\n'.format(
                    C1 / C2))
        else:
            print(
                '   tDCF_norm(s) = Pmiss_cm(s) + {:8.5f} x Pfa_cm(s)\n'.format(
                    C2 / C1))

    return tDCF_norm, CM_thresholds


def calculate_CLLR(target_llrs, nontarget_llrs):
    """
    Calculate the CLLR of the scores.
    
    Parameters:
    target_llrs (list or numpy array): Log-likelihood ratios for target trials.
    nontarget_llrs (list or numpy array): Log-likelihood ratios for non-target trials.
    
    Returns:
    float: The calculated CLLR value.
    """
    def negative_log_sigmoid(lodds):
        """
        Calculate the negative log of the sigmoid function.
        
        Parameters:
        lodds (numpy array): Log-odds values.
        
        Returns:
        numpy array: The negative log of the sigmoid values.
        """
        return np.log1p(np.exp(-lodds))

    # Convert the input lists to numpy arrays if they are not already
    target_llrs = np.array(target_llrs)
    nontarget_llrs = np.array(nontarget_llrs)
    
    # Calculate the CLLR value
    cllr = 0.5 * (np.mean(negative_log_sigmoid(target_llrs)) + np.mean(negative_log_sigmoid(-nontarget_llrs))) / np.log(2)
    
    return cllr


def compute_Pmiss_Pfa_Pspoof_curves(tar_scores, non_scores, spf_scores):

    # Concatenate all scores and designate arbitrary labels 1=target, 0=nontarget, -1=spoof
    all_scores = np.concatenate((tar_scores, non_scores, spf_scores))
    labels = np.concatenate((np.ones(tar_scores.size), np.zeros(non_scores.size), -1*np.ones(spf_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Cumulative sums
    tar_sums    = np.cumsum(labels==1)
    non_sums    = np.cumsum(labels==0)
    spoof_sums  = np.cumsum(labels==-1)

    Pmiss       = np.concatenate((np.atleast_1d(0), tar_sums / tar_scores.size))
    Pfa_non     = np.concatenate((np.atleast_1d(1), 1 - (non_sums / non_scores.size)))
    Pfa_spoof   = np.concatenate((np.atleast_1d(1), 1 - (spoof_sums / spf_scores.size)))
    thresholds  = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores

    return Pmiss, Pfa_non, Pfa_spoof, thresholds


def compute_teer(Pmiss_CM, Pfa_CM, tau_CM, Pmiss_ASV, Pfa_non_ASV, Pfa_spf_ASV, tau_ASV):
    # Different spoofing prevalence priors (rho) parameters values
    rho_vals            = [0,0.5,1]

    tEER_val    = np.empty([len(rho_vals),len(tau_ASV)], dtype=float)

    for rho_idx, rho_spf in enumerate(rho_vals):

        # Table to store the CM threshold index, per each of the ASV operating points
        tEER_idx_CM = np.empty(len(tau_ASV), dtype=int)

        tEER_path   = np.empty([len(rho_vals),len(tau_ASV),2], dtype=float)

        # Tables to store the t-EER, total Pfa and total miss valuees along the t-EER path
        Pmiss_total = np.empty(len(tau_ASV), dtype=float)
        Pfa_total   = np.empty(len(tau_ASV), dtype=float)
        min_tEER    = np.inf
        argmin_tEER = np.empty(2)

        # best intersection point
        xpoint_crit_best = np.inf
        xpoint = np.empty(2)

        # Loop over all possible ASV thresholds
        for tau_ASV_idx, tau_ASV_val in enumerate(tau_ASV):

            # Tandem miss and fa rates as defined in the manuscript
            Pmiss_tdm = Pmiss_CM + (1 - Pmiss_CM) * Pmiss_ASV[tau_ASV_idx]
            Pfa_tdm   = (1 - rho_spf) * (1 - Pmiss_CM) * Pfa_non_ASV[tau_ASV_idx] + rho_spf * Pfa_CM * Pfa_spf_ASV[tau_ASV_idx]

            # Store only the INDEX of the CM threshold (for the current ASV threshold)
            h = Pmiss_tdm - Pfa_tdm
            tmp = np.argmin(abs(h))
            tEER_idx_CM[tau_ASV_idx] = tmp

            if Pmiss_ASV[tau_ASV_idx] < (1 - rho_spf) * Pfa_non_ASV[tau_ASV_idx] + rho_spf * Pfa_spf_ASV[tau_ASV_idx]:
                Pmiss_total[tau_ASV_idx] = Pmiss_tdm[tmp]
                Pfa_total[tau_ASV_idx] = Pfa_tdm[tmp]

                tEER_val[rho_idx,tau_ASV_idx] = np.mean([Pfa_total[tau_ASV_idx], Pmiss_total[tau_ASV_idx]])

                tEER_path[rho_idx,tau_ASV_idx, 0] = tau_ASV_val
                tEER_path[rho_idx,tau_ASV_idx, 1] = tau_CM[tmp]

                if tEER_val[rho_idx,tau_ASV_idx] < min_tEER:
                    min_tEER = tEER_val[rho_idx,tau_ASV_idx]
                    argmin_tEER[0] = tau_ASV_val
                    argmin_tEER[1] = tau_CM[tmp]

                # Check how close we are to the INTERSECTION POINT for different prior (rho) values:
                LHS = Pfa_non_ASV[tau_ASV_idx]/Pfa_spf_ASV[tau_ASV_idx]
                RHS = Pfa_CM[tmp]/(1 - Pmiss_CM[tmp])
                crit = abs(LHS - RHS)

                if crit < xpoint_crit_best:
                    xpoint_crit_best = crit
                    xpoint[0] = tau_ASV_val
                    xpoint[1] = tau_CM[tmp]
                    xpoint_tEER = Pfa_spf_ASV[tau_ASV_idx]*Pfa_CM[tmp]
            else:
                # Not in allowed region
                tEER_path[rho_idx,tau_ASV_idx, 0] = np.nan
                tEER_path[rho_idx,tau_ASV_idx, 1] = np.nan
                Pmiss_total[tau_ASV_idx] = np.nan
                Pfa_total[tau_ASV_idx] = np.nan
                tEER_val[rho_idx,tau_ASV_idx] = np.nan

        return xpoint_tEER*100
