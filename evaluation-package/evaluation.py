import argparse
import sys

from calculate_metrics import calculate_minDCF_EER_CLLR, calculate_aDCF_tdcf_tEER

def main(args: argparse.Namespace) -> None:

    if args.mode == "p1":
        eval_eer, eval_dcf, eval_cllr = calculate_minDCF_EER_CLLR(
                    cm_scores_file=args.score_cm,
                    output_file="./phase1_result.txt")
        print("# Phase 1 Result: \n")
        print("-eval_eer: {:.3f}\n-eval_dcf:{:.5f}\n-eval_cllr:{:.5f}\n".format(eval_eer, eval_dcf, eval_cllr))
        sys.exit(0)
        
    elif args.mode == "p2_tandem":
        if len(sys.argv) > 2:
            eval_adcf, eval_tdcf, eval_teer = calculate_aDCF_tdcf_tEER(
                        cm_scores_file=args.score_cm,
                        asv_scores_file= args.score_asv,
                        output_file="./phase2_result.txt")
            print("# Phase 2 Result: \n")
            print("-eval_adcf: {:.3f}\n-eval_tdcf:{:.5f}\n-eval_teer:{:.5f}\n".format(eval_adcf, eval_tdcf, eval_teer))
            sys.exit(0)

    elif args.mode == "p2_single":
        from a_dcf import a_dcf
        adcf = a_dcf.calculate_a_dcf(args.score_cm)['min_a_dcf']
        print("# Phase 2 (Single) Result: \n")
        print("-eval_adcf: {:.3f}\n".format(eval_adcf))
        sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m",
                        dest="mode",
                        type=str,
                        help="mode flag: p1(Phase 1) or p2_tandem(Phaes 2) or p2_single(Phaes 2)",
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

    main(parser.parse_args())