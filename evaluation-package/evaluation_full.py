#!/usr/bin/env python3
"""
main function to run evaluation package using
the full meta key table

"""
import sys
import argparse
import numpy as np

from calculate_metrics_full import calculate_minDCF_EER_CLLR_actDCF_full
from calculate_metrics_full import calculate_aDCF_tdcf_tEER_full
import util
import util_table

def main(args: argparse.Namespace) -> None:

    if args.mode == "t1":

        # load the score and key table
        cm_score_key_pd = util.load_cm_scores_keys_as_pd(args.score_cm, args.key_cm)

        # get results
        attack_list, codec_list, minDCF_mat, eer_mat, cllr_mat, actDCF_mat = calculate_minDCF_EER_CLLR_actDCF_full(cm_score_key_pd)

        # print output results
        print("# Track 1 Result: \neval_minDCF\n")
        _ = util_table.print_table(minDCF_mat, codec_list, attack_list, '1.5f',
                                   with_color_cell=args.flag_latex_color);
        print('\nEER (%)\n')
        _ = util_table.print_table(eer_mat * 100, codec_list, attack_list, '1.3f',
                                   with_color_cell=args.flag_latex_color);
        print('\nCllr bits\n')
        _ = util_table.print_table(cllr_mat, codec_list, attack_list, '1.5f',
                                   with_color_cell=args.flag_latex_color);   
        print('\nactDCF\n')
        _ = util_table.print_table(actDCF_mat, codec_list, attack_list, '1.5f',
                                   with_color_cell=args.flag_latex_color);        
        sys.exit(0)
        
    elif args.mode == "t2_tandem":
        
        # load the score and key table
        sasv_score_key_pd = util.load_sasv_scores_keys_as_pd(args.score_sasv, args.key_sasv)
        
        # load the official asv key and scores
        asv_org_score_key_pd = util.load_sasv_scores_keys_as_pd(args.score_asv, args.key_sasv)

        # compute results
        attack_list, codec_list, adcf_mat, min_tDCF_mat, teer_mat = calculate_aDCF_tdcf_tEER_full(sasv_score_key_pd, asv_org_score_key_pd)
        
        # print output results
        print("# Track 2 Result: \na-DCF\n")
        _ = util_table.print_table(adcf_mat, codec_list, attack_list, '1.5f',
                                   with_color_cell=args.flag_latex_color);
        print('\nmin tDCF\n')
        _ = util_table.print_table(min_tDCF_mat, codec_list, attack_list, '1.5f',
                                   with_color_cell=args.flag_latex_color);
        print('\nt-EER (%)\n')
        _ = util_table.print_table(teer_mat, codec_list, attack_list, '1.3f',
                                   with_color_cell=args.flag_latex_color);   

        sys.exit(0)

    elif args.mode == "t2_single":
        # load the score and key table
        sasv_score_key_pd = util.load_sasv_scores_keys_as_pd(args.score_sasv, args.key_sasv)
        
        # compute results
        attack_list, codec_list, adcf_mat, _, _ = calculate_aDCF_tdcf_tEER_full(sasv_score_key_pd, None, flag_sasv_only=True)
        
        # print output results
        print("# Track 2 Result: \na-DCF\n")
        _ = util_table.print_table(adcf_mat, codec_list, attack_list, '1.5f',
                                   with_color_cell=args.flag_latex_color);
   
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


    parser.add_argument("--asv_org",
                        dest="score_asv",
                        type=str,
                        help="asv scores from organizer")

    parser.add_argument('--flag_latex_table_w_color',
                        dest='flag_latex_color',
                        action='store_true',
                        help="whether add cell color to latex table")

    
    main(parser.parse_args())
