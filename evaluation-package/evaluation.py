import argparse
import sys

from calculate_metrics import calculate_minDCF_EER_CLLR, calculate_aDCF_tdcf_tEER

def main(args: argparse.Namespace) -> None:

    if args.mode == "t1":
        minDCF, eer, cllr = calculate_minDCF_EER_CLLR(
                    cm_scores_file=args.score_cm,
                    output_file="./phase1_result.txt")
        print("# Track 1 Result: \n")
        print("-eval_eer: {:.3f}\n-eval_dcf:{:.5f}\n-eval_cllr:{:.5f}\n".format(eer*100, minDCF, cllr*100))
        sys.exit(0)
        
    elif args.mode == "t2_tandem":
        if len(sys.argv) > 2:
            adcf, min_tDCF, teer = calculate_aDCF_tdcf_tEER(
                        cm_scores_file=args.score_cm,
                        asv_scores_file= args.score_asv,
                        output_file="./phase2_result.txt")
            print("# Track 2 Result: \n")
            print("-eval_adcf: {:.3f}\n-eval_tdcf:{:.5f}\n-eval_teer:{:.5f}\n".format(adcf, min_tDCF, teer))
            sys.exit(0)

    elif args.mode == "t2_single":
        from a_dcf import a_dcf
        adcf = a_dcf.calculate_a_dcf(args.score_sasv)['min_a_dcf']
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
                        help="cm score file as input",
                        required=True)
    
    parser.add_argument("--asv",
                        dest="score_asv",
                        type=str,
                        help="asv score as input")
    
    parser.add_argument("--sasv",
                        dest="score_sasv",
                        type=str,
                        help="sasv score as input")

    main(parser.parse_args())