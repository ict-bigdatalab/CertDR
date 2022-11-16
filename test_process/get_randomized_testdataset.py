import os, random, pickle, sys
from os.path import abspath, join, dirname
sys.path.insert(0, join(abspath(dirname(__file__)), '..'))

import numpy as np
import argparse
from tqdm import tqdm

import torch

from test_process.tv_table import calculate_tv_table
from word_sub_helper import WordSubstitude
from utils import read_qd_pairs, read_qdr_pairs


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def randomize_testset(args, perturb_pca, random_smooth, similarity_threshold, perturbation_constraint):
    data_dir = args.data_dir
    num_random_sample = args.num_random_sample

    dataset_name = args.task_name

    sampled_qd = read_qd_pairs(args.sampled_qd_path)
    qdr = read_qdr_pairs(args.bert_top100_ranked_list_path)

    example_qid_list = list(qdr.keys())[:200]

    K = args.certify_K
    doc_list = []

    for example_qid in example_qid_list:
        ranked_docs = qdr[example_qid]
        K_docid = ''
        for rd in ranked_docs:
            if ranked_docs[rd] == K:
                K_docid = rd
            elif ranked_docs[rd] < K:
                continue
            else:
                if rd not in doc_list:
                    doc_list.append(rd)

        if K_docid not in doc_list:
            doc_list.append(K_docid)

    out_data_dir = os.path.join(data_dir,
                                'random_test_docs' + str(similarity_threshold) + '_' + str(perturbation_constraint))
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)
    if not os.path.exists(out_data_dir + '/tokenized'):
        os.makedirs(out_data_dir + '/tokenized')
    print('Generating randomized data for document')

    if dataset_name == 'msmarco-doc':
        from marcodoc.dataset import CollectionDataset
        collection = CollectionDataset(args.collection_memmap_dir)

    elif dataset_name == 'msmarco-pas':
        from marcopas.dataset import CollectionDataset
        collection = CollectionDataset(args.collection_memmap_dir)
    else:
        raise ValueError('check the dataset')

    from transformers import BertTokenizer
    bert_tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer_dir)

    for did in tqdm(doc_list):
        cached_doc_file = os.path.join(out_data_dir, str(did) + '_num_' + str(num_random_sample))

        if os.path.exists(cached_doc_file):
            tokenized_ids_path = out_data_dir + '/tokenized/' + str(did) + '_num_' \
                                 + str(num_random_sample)
            if os.path.exists(tokenized_ids_path):
                continue
            else:

                doc_perturb_list = []
                with open(cached_doc_file, 'r') as rf:
                    lines = rf.readlines()[1:]  # skip the first line.
                    for line in lines:
                        doc_perturb_list.append(line.strip())

                assert len(doc_perturb_list) == num_random_sample

        else:
            wf = open(cached_doc_file, 'w')
            ori_doc_tokens = collection[did]
            ori_doc = bert_tokenizer.decode(ori_doc_tokens)

            wf.write(ori_doc + '\n')
            doc_perturb_list = []
            for _ in range(num_random_sample):
                doc_perturb = str(random_smooth.get_perturbed_batch(np.array([[ori_doc]]))[0][0])
                doc_perturb_list.append(doc_perturb)
                wf.write(doc_perturb + '\n')
            wf.close()

    tv_table_dir = os.path.join(data_dir, dataset_name + '_counterfitted_tv_pca' +
                                str(similarity_threshold) + '_' + str(perturbation_constraint) + '.pkl')

    if not os.path.exists(tv_table_dir):
        tv_table = calculate_tv_table(args, perturb_pca)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " )

    parser.add_argument("--num_random_sample", default=2000, type=int,
                        help="The number of random samples of each text.")
    parser.add_argument("--similarity_threshold", default=0.5, type=float,
                        help="The similarity constraint to be considered as synonym.")
    parser.add_argument("--perturbation_constraint", default=100, type=int,
                        help="The maximum size of perturbation set of each word")
    parser.add_argument("--sampled_qd_path",
                        default='',
                        type=str, help="")
    parser.add_argument("--bert_top100_ranked_list_path",
                        default='',
                        type=str, help="")
    parser.add_argument("--certify_K", default=20,
                        type=int, help="the top K document that you want to certify (cannot be ranked beyond it)")

    # Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--bert_tokenizer_dir", default="", type=str, help="")
    parser.add_argument("--collection_memmap_dir", default="", type=str, help="")



    args = parser.parse_args()

    dataset_name = args.task_name
    similarity_threshold = args.similarity_threshold
    perturbation_constraint = args.perturbation_constraint

    pkl_file = open(
        args.data_dir + dataset_name + '_perturbation_constraint_pca' + str(similarity_threshold) + '_' + str(
            perturbation_constraint) + '.pkl', 'rb')
    perturb_pca = pickle.load(pkl_file)
    pkl_file.close()

    for key in perturb_pca.keys():
        if len(perturb_pca[key]['set']) > perturbation_constraint:

            tem_neighbor_count = 0
            tem_neighbor_list = []
            for tem_neighbor in perturb_pca[key]['set']:
                tem_neighbor_list.append(tem_neighbor)
                tem_neighbor_count += 1
                if tem_neighbor_count >= perturbation_constraint:
                    break
            perturb_pca[key]['set'] = tem_neighbor_list
            perturb_pca[key]['isdivide'] = 1

    random_smooth = WordSubstitude(perturb_pca)
    randomize_testset(args, perturb_pca, random_smooth, similarity_threshold, perturbation_constraint)
    print('finished.')


if __name__ == '__main__':
    main()