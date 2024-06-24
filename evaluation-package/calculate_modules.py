import sys
import numpy as np


def obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold):
    """
    compute ASV error rates
    
    input
    -----
      tar_asv: np.array, (#N, ), target bonafide scores
      non_asv: np.array, (#M, ), nontarget bonafide scores
      spoof_asv: np.array, (#K, ), spoof scores
      asv_threshold: scalar, ASV threshold

    output
    ------
      Pfa_asv: scalar, false acceptance rate of nontarget bonafide
      Pmiss_asv: scalar, miss rate of target bonafide
      Pmiss_spoof_asv: scalar, 1 - Pfa_spoof_asv
      Pfa_spoof_asv: scalar, false acceptance rate of spoofed data
    """
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
    """
    compute DET curve values
                                                                           
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
    


def compute_tDCF_legacy(
        bonafide_score_cm, spoof_score_cm, Pfa_asv, Pmiss_asv,
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


def compute_tDCF(
        bonafide_score_cm, spoof_score_cm, 
        Pfa_asv, Pmiss_asv, Pfa_spoof_asv, cost_model, print_cost):
    """
    Compute Tandem Detection Cost Function (t-DCF) [1] for a fixed ASV system.
    In brief, t-DCF returns a detection cost of a cascaded system of this form,

      Speech waveform -> [CM] -> [ASV] -> decision

    where CM stands for countermeasure and ASV for automatic speaker
    verification. The CM is therefore used as a 'gate' to decided whether or
    not the input speech sample should be passed onwards to the ASV system.
    Generally, both CM and ASV can do detection errors. Not all those errors
    are necessarily equally cost, and not all types of users are necessarily
    equally likely. The tandem t-DCF gives a principled with to compare
    different spoofing countermeasures under a detection cost function
    framework that takes that information into account.

    INPUTS:

      bonafide_score_cm   A vector of POSITIVE CLASS (bona fide or human)
                          detection scores obtained by executing a spoofing
                          countermeasure (CM) on some positive evaluation trials.
                          trial represents a bona fide case.
      spoof_score_cm      A vector of NEGATIVE CLASS (spoofing attack)
                          detection scores obtained by executing a spoofing
                          CM on some negative evaluation trials.
      Pfa_asv             False alarm (false acceptance) rate of the ASV
                          system that is evaluated in tandem with the CM.
                          Assumed to be in fractions, not percentages.
      Pmiss_asv           Miss (false rejection) rate of the ASV system that
                          is evaluated in tandem with the spoofing CM.
                          Assumed to be in fractions, not percentages.
      Pmiss_spoof_asv     Miss rate of spoof samples of the ASV system that
                          is evaluated in tandem with the spoofing CM. That
                          is, the fraction of spoof samples that were
                          rejected by the ASV system.
      cost_model          A struct that contains the parameters of t-DCF,
                          with the following fields.

                          Ptar        Prior probability of target speaker.
                          Pnon        Prior probability of nontarget speaker (zero-effort impostor)
                          Psoof       Prior probability of spoofing attack.
                          Cmiss       Cost of tandem system falsely rejecting target speaker.
                          Cfa         Cost of tandem system falsely accepting nontarget speaker.
                          Cfa_spoof   Cost of tandem system falsely accepting spoof.

      print_cost          Print a summary of the cost parameters and the
                          implied t-DCF cost function?

    OUTPUTS:

      tDCF_norm           Normalized t-DCF curve across the different CM
                          system operating points; see [2] for more details.
                          Normalized t-DCF > 1 indicates a useless
                          countermeasure (as the tandem system would do
                          better without it). min(tDCF_norm) will be the
                          minimum t-DCF used in ASVspoof 2019 [2].
      CM_thresholds       Vector of same size as tDCF_norm corresponding to
                          the CM threshold (operating point).

    NOTE:
    o     In relative terms, higher detection scores values are assumed to
          indicate stronger support for the bona fide hypothesis.
    o     You should provide real-valued soft scores, NOT hard decisions. The
          recommendation is that the scores are log-likelihood ratios (LLRs)
          from a bonafide-vs-spoof hypothesis based on some statistical model.
          This, however, is NOT required. The scores can have arbitrary range
          and scaling.
    o     Pfa_asv, Pmiss_asv, Pmiss_spoof_asv are in fractions, not percentages.

    References:

      [1] T. Kinnunen, H. Delgado, N. Evans,K.-A. Lee, V. Vestman, 
          A. Nautsch, M. Todisco, X. Wang, M. Sahidullah, J. Yamagishi, 
          and D.-A. Reynolds, "Tandem Assessment of Spoofing Countermeasures
          and Automatic Speaker Verification: Fundamentals," IEEE/ACM Transaction on
          Audio, Speech and Language Processing (TASLP).

      [2] ASVspoof 2019 challenge evaluation plan
          https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf
    """


    # Sanity check of cost parameters
    if cost_model['Cfa'] < 0 or cost_model['Cmiss'] < 0 or \
            cost_model['Cfa'] < 0 or cost_model['Cmiss'] < 0:
        print('WARNING: Usually the cost values should be positive!')

    if cost_model['Ptar'] < 0 or cost_model['Pnon'] < 0 or cost_model['Pspoof'] < 0 or \
            np.abs(cost_model['Ptar'] + cost_model['Pnon'] + cost_model['Pspoof'] - 1) > 1e-10:
        sys.exit('ERROR: Your prior probabilities should be positive and sum up to one.')

    # Unless we evaluate worst-case model, we need to have some spoof tests against asv
    if Pfa_spoof_asv is None:
        sys.exit('ERROR: you should provide false alarm rate of spoof tests against your ASV system.')

    # Sanity check of scores
    combined_scores = np.concatenate((bonafide_score_cm, spoof_score_cm))
    if np.isnan(combined_scores).any() or np.isinf(combined_scores).any():
        sys.exit('ERROR: Your scores contain nan or inf.')

    # Sanity check that inputs are scores and not decisions
    n_uniq = np.unique(combined_scores).size
    if n_uniq < 3:
        sys.exit('ERROR: You should provide soft CM scores - not binary decisions')

    # Obtain miss and false alarm rates of CM
    Pmiss_cm, Pfa_cm, CM_thresholds = compute_det_curve(bonafide_score_cm, spoof_score_cm)

    # Constants - see ASVspoof 2019 evaluation plan

    C0 = cost_model['Ptar'] * cost_model['Cmiss'] * Pmiss_asv + cost_model['Pnon']*cost_model['Cfa']*Pfa_asv
    C1 = cost_model['Ptar'] * cost_model['Cmiss'] - (cost_model['Ptar'] * cost_model['Cmiss'] * Pmiss_asv + cost_model['Pnon'] * cost_model['Cfa'] * Pfa_asv)
    C2 = cost_model['Pspoof'] * cost_model['Cfa_spoof'] * Pfa_spoof_asv;


    # Sanity check of the weights
    if C0 < 0 or C1 < 0 or C2 < 0:
        sys.exit('You should never see this error but I cannot evalute tDCF with negative weights - please check whether your ASV error rates are correctly computed?')

    # Obtain t-DCF curve for all thresholds
    tDCF = C0 + C1 * Pmiss_cm + C2 * Pfa_cm

    # Obtain default t-DCF
    tDCF_default = C0 + np.minimum(C1, C2)

    # Normalized t-DCF
    tDCF_norm = tDCF / tDCF_default

    # Everything should be fine if reaching here.
    if print_cost:

        print('t-DCF evaluation from [Nbona={}, Nspoof={}] trials\n'.format(bonafide_score_cm.size, spoof_score_cm.size))
        print('t-DCF MODEL')
        print('   Ptar         = {:8.5f} (Prior probability of target user)'.format(cost_model['Ptar']))
        print('   Pnon         = {:8.5f} (Prior probability of nontarget user)'.format(cost_model['Pnon']))
        print('   Pspoof       = {:8.5f} (Prior probability of spoofing attack)'.format(cost_model['Pspoof']))
        print('   Cfa          = {:8.5f} (Cost of tandem system falsely accepting a nontarget)'.format(cost_model['Cfa']))
        print('   Cmiss        = {:8.5f} (Cost of tandem system falsely rejecting target speaker)'.format(cost_model['Cmiss']))
        print('   Cfa_spoof    = {:8.5f} (Cost of tandem sysmte falsely accepting spoof)'.format(cost_model['Cfa_spoof']))
        print('\n   Implied normalized t-DCF function (depends on t-DCF parameters and ASV errors), t_CM=CM threshold)')
        print('   tDCF_norm(t_CM) = {:8.5f} + {:8.5f} x Pmiss_cm(t_CM) + {:8.5f} x Pfa_cm(t_CM)\n'.format(C0/tDCF_default, C1/tDCF_default, C2/tDCF_default))
        print('     * The optimum value is given by the first term (0.06273). This is the normalized t-DCF obtained with an error-free CM system.')
        print('     * The minimum normalized cost (minimum over all possible thresholds) is always <= 1.00.')
        print('')

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



def compute_teer(Pmiss_CM, Pfa_CM, tau_CM, Pmiss_ASV, Pfa_non_ASV, Pfa_spf_ASV, tau_ASV):
    """Compute concurrent t-EER
    
    input
    -----
      Pmiss_CM: np.array, (N, ), miss rates of CM
      Pfa_CM: np.array, (N, ), false acceptanace rates of CM
      tau_cm: np.array, (N, ), thresholds of CM

      Pmiss_ASV: np.array, (M, ), miss rates of ASV
      Pfa_non_ASV: np.array, (M, ), false acc rates of nontarget data ASV
      Pfa_spf_ASV: np.array, (M, ), false acc rates of spoofed data of ASV
      tau_asv: np.array, (M, ), thresholds of ASV
    
    output
    ------
      con_t-eer: scalar, concurrent EER

    Note that N = len(bonafide_CM_scores) + len(spoofed_CM_scores)
    M = len(target_ASV_scores) + len(nontarget_ASV_scores) + len(spoofed)
    """

    
    # Different spoofing prevalence priors (rho) parameters values
    # rho_vals            = [0,0.5,1]
    # for concurrent t-EER, any rho gives the same result, so use one
    rho_vals = [0.5]

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
