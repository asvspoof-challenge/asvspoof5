import argparse
import sys

from calculate_metrics import calculate_minDCF_EER_CLLR, calculate_aDCF_tdcf_tEER
import a_dcf
import util

def main(args: argparse.Namespace) -> None:

    if args.mode == "t1":

        # load score and keys
        cm_scores, cm_keys = util.load_cm_scores_keys(args.score_cm, args.key_cm)
        
        minDCF, eer, cllr, actDCF = calculate_minDCF_EER_CLLR(
            cm_scores = cm_scores,
            cm_keys = cm_keys,
            output_file="./phase1_result.txt")
        print("# Track 1 Result: \n")
        print("-eval_mindcf:{:.5f}\n-eval_eer (%): {:.3f}\n-eval_cllr (bits):{:.5f}\n-eval_actDCF:{:.5f}\n".format(
            minDCF, eer*100, cllr, actDCF))
        sys.exit(0)
        
    elif args.mode == "t2_tandem":
        if len(sys.argv) > 2:

            # load score and keys
            cm_scores, asv_scores, sasv_scores, cm_keys, asv_keys = util.load_sasv_scores_keys(
                args.score_sasv, args.key_sasv)
        
            adcf, min_tDCF, teer = calculate_aDCF_tdcf_tEER(
                cm_scores = cm_scores,
                asv_scores = asv_scores,
                sasv_scores = sasv_scores,
                cm_keys = cm_keys,
                asv_keys = asv_keys,
                output_file="./phase2_result.txt")
            print("# Track 2 Result: \n")
            print("-eval_adcf: {:.3f}\n-eval_tdcf:{:.5f}\n-eval_teer:{:.5f}\n".format(adcf, min_tDCF, teer))
            sys.exit(0)

    elif args.mode == "t2_single":
        # load score and keys
        _, _, sasv_scores, _, asv_keys = util.load_sasv_scores_keys(
            args.score_sasv, args.key_sasv)

        # Extract SASV scores
        trg_scores = sasv_scores[asv_keys == 'target']
        nontrg_scores = sasv_scores[asv_keys == 'nontarget']
        spoof_scores = sasv_scores[asv_keys == 'spoof']

        # compute a-DCF
        adcf = a_dcf._calculate_a_dcf(trg_scores, nontrg_scores, spoof_scores)['min_a_dcf']
        print("# Track 2 (Single) Result: \n")
        print("-eval_adcf: {:.3f}\n".format(adcf))
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
