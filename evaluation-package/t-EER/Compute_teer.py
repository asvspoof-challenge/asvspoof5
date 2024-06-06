import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy import special
import random
import sys
import os
import eval_metrics as em


args = sys.argv

# Use CM scores file path
cm_score_file = args[1] # CM scores file for ASVspoof5.eval.trail.txt
# Use ASV scores file path
asv_score_file = args[2] # ASV scores file for ASVspoof5.eval.trail.txt
# give path for groundtruth 
label_list = args[3]


X_tar, X_non, X_spf = em.process_ASVspoof5_score_files(cm_score_file, asv_score_file,label_list)

# Obtain ASV error curves and ASV thresholds
Pmiss_ASV, Pfa_non_ASV, Pfa_spf_ASV, tau_ASV = em.compute_Pmiss_Pfa_Pspoof_curves(X_tar[0], X_non[0], X_spf[0])

# Obtain CM error curves and CM thresholds.
Pmiss_CM, Pfa_CM, tau_CM = em.compute_det_curve(np.concatenate([X_tar[1], X_non[1]]), X_spf[1])


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


    print("concurrent-teer for [rho :{}] = {:.2f} ".format(rho_spf,xpoint_tEER*100))
