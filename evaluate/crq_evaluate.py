import os, pickle, math, argparse
import numpy as np
import random
from tqdm import tqdm

from test_process.tv_table import Calculate_tv
from utils import read_qds_pairs


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def get_crq(f_RS_score_path, data_dir, dataset_name, similarity_threshold,
            perturbation_constraint, num_samples=1000, consider_mc_error=True):

    print(f_RS_score_path)
    mc_error = 2 * math.sqrt(math.log(2 / 0.05)/(2 * num_samples))
    print('mc_error:', mc_error)
    # certify_K = 10

    certify_Ks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    default_seed = 2022
    set_seed(default_seed)

    tv_table_dir = os.path.join(data_dir, dataset_name + '_counterfitted_tv_pca' + str(similarity_threshold) + '_' + str(
                                perturbation_constraint) + '.pkl')

    print(tv_table_dir)
    pkl_file = open(tv_table_dir, 'rb')
    tv_table = pickle.load(pkl_file)
    pkl_file.close()

    tv = Calculate_tv(tv_table)
    f_RS_qds = read_qds_pairs(f_RS_score_path)
    qdr = {}
    for qid, docs in f_RS_qds.items():
        ranked = sorted(docs, key=docs.get, reverse=True)
        if qid not in qdr:
            qdr[qid] = {}
        index = 1
        for did in ranked:
            qdr[qid][did] = index
            index += 1
    for certify_K in certify_Ks:
        cr_q_list = []
        for qid in tqdm(qdr):

            ranked_docs = qdr[qid]

            test_doc_list = []
            K = certify_K
            K_docid = ''
            for rd in ranked_docs:
                if ranked_docs[rd] == K:
                    K_docid = rd
                if ranked_docs[rd] > K:
                    test_doc_list.append(rd)

            random_doc_dir = os.path.join(data_dir, 'random_test_docs' + str(similarity_threshold) + '_' +
                                          str(perturbation_constraint) + '_without_tokenize')
            estimated_K_score = f_RS_qds[int(qid)][int(K_docid)]

            assert os.path.exists(random_doc_dir)

            max_o_d = -1
            for test_doc_id in test_doc_list:
                did = test_doc_id

                test_doc_path = str(did) + '_num_' + str(num_samples)

                rf = open(random_doc_dir + '/' + test_doc_path, 'r')
                ori_doc_content = rf.readline().strip()
                rf.close()

                tem_tv = tv.get_tv(ori_doc_content)
                truncated_tem_tv = tem_tv
                o_d = 1. - np.prod(truncated_tem_tv)
                if o_d > max_o_d:
                    max_o_d = o_d

                if ranked_docs[int(did)] == (K + 1):
                    estimated_K_plus_one_score = f_RS_qds[int(qid)][int(did)]

            delta_x_gu = estimated_K_score - estimated_K_plus_one_score - max_o_d

            if consider_mc_error:
                if delta_x_gu - mc_error > 0:
                    certified_robust = 1
                else:
                    certified_robust = 0
            else:

                if delta_x_gu > 0:
                    certified_robust = 1
                else:
                    certified_robust = 0

            cr_q_list.append(certified_robust)

        print('K:', certify_K)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--similarity_threshold", default=0.9, type=float,
                        help="The similarity constraint to be considered as synonym.")
    parser.add_argument("--perturbation_constraint", default=100, type=int,
                        help="The maximum size of perturbation set of each word")
    parser.add_argument("--data_dir", default='/data/msmarco-doc/', type=str,
                        help="Your dataset prefix")
    parser.add_argument("--dataset_name", default='msmarco-doc', type=str,
                        help="Your dataset name")
    parser.add_argument("--f_RS_score_path", default='', type=str,
                        help="fRS score path")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--consider_mc_error", action='store_true',
                        help="Whether to consider the error of the Monte Carlo estimation")
    args = parser.parse_args()

    get_crq(args.f_RS_score_path, args.data_dir, args.dataset_name, args.similarity_threshold,
            args.perturbation_constraint, num_samples=args.num_samples, consider_mc_error=args.consider_mc_error)


if __name__ == "__main__":
    main()