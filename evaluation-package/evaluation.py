#!/usr/bin/env python3
"""
main function to run evaluation package.

See usage in README.md
"""
import argparse
import sys

from calculate_metrics import calculate_minDCF_EER_CLLR
from calculate_metrics import calculate_aDCF_tdcf_tEER
import util

def main(args: argparse.Namespace) -> None:

    if args.mode == "t1":
        
        # load score and keys
        cm_scores, cm_keys = util.load_cm_scores_keys(args.score_cm, args.key_cm)
        
        minDCF, eer, cllr, actDCF = calculate_minDCF_EER_CLLR(
            cm_scores = cm_scores,
            cm_keys = cm_keys,
            output_file="./track1_result.txt")
        print("# Track 1 Result: \n")
        print("-eval_mindcf: {:.5f}\n-eval_eer (%): {:.3f}\n-eval_cllr (bits): {:.5f}\n-eval_actDCF: {:.5f}\n".format(
            minDCF, eer*100, cllr, actDCF))
        sys.exit(0)
        
    elif args.mode == "t2_tandem":
        # load score and keys
        cm_scores, asv_scores, sasv_scores, cm_keys, asv_keys = util.load_sasv_scores_keys(args.score_sasv, args.key_sasv)
        
        adcf, min_tDCF, teer = calculate_aDCF_tdcf_tEER(
            cm_scores = cm_scores,
            asv_scores = asv_scores,
            sasv_scores = sasv_scores,
            cm_keys = cm_keys,
            asv_keys = asv_keys,
            output_file="./track2_result.txt")
        print("# Track 2 Result: \n")
        print("-eval_adcf: {:.5f}\n-eval_tdcf: {:.5f}\n-eval_teer (%): {:.3f}\n".format(adcf, min_tDCF, teer))
        sys.exit(0)

    elif args.mode == "t2_single":
        # load score and keys
        _, _, sasv_scores, _, asv_keys = util.load_sasv_scores_keys(
            args.score_sasv, args.key_sasv)

        adcf = calculate_aDCF_tdcf_tEER(
            cm_scores = None,
            asv_scores = None,
            sasv_scores = sasv_scores,
            cm_keys = None,
            asv_keys = asv_keys,
            output_file="./track2_result_adcf_only.txt")
        print("# Track 2 (Single) Result: \n")
        print("-eval_adcf: {:.5f}\n".format(adcf))
        sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m",
                        dest="mode",
                        type=str,
                        help="mode flag: t1(Track 1) or t2_tandem(Track 2) or t2_single(Track 2)",
                        required=True)
    
    parser.add_argument("--cm",
                        dest="score_cm",
                        type=str,
                        help="cm score file as input")

    parser.add_argument("--cm_keys",
                        dest="key_cm",
                        type=str,
                        help="cm key file as input")
    
    parser.add_argument("--sasv",
                        dest="score_sasv",
                        type=str,
                        help="sasv score as input")
    
    parser.add_argument("--sasv_keys",
                        dest="key_sasv",
                        type=str,
                        help="sasv key as input")

    main(parser.parse_args())
