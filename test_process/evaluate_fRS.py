from __future__ import absolute_import, division, print_function

import sys
from os.path import abspath, join, dirname
sys.path.insert(0, join(abspath(dirname(__file__)), '..'))

import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)
from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertConfig, BertTokenizer)

from modeling import Ranking_BERT_Train
from marcodoc.dataset import MSMARCORandomDocTestDataset, get_collate_function_randomized_test
from utils import read_qdr_pairs, read_qds_pairs
from marcodoc.dataset import load_queries

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, Ranking_BERT_Train, BertTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def pack_tensor_2D(lstlst, default, dtype, length=None):
    batch_size = len(lstlst)
    length = length if length is not None else max(len(l) for l in lstlst)
    tensor = default * torch.ones((batch_size, length), dtype=dtype)
    for i, l in enumerate(lstlst):
        # print(i, l)
        tensor[i, :len(l)] = torch.tensor(l, dtype=dtype)
    return tensor


def eval_model(model, doc_input_ids, query_input_ids, args, cls_id, sep_id, max_query_length=64, max_doc_length=445):
    query_input_ids = query_input_ids[:max_query_length]
    query_input_ids = [cls_id] + query_input_ids + [sep_id]
    doc_input_ids = doc_input_ids[:max_doc_length]
    doc_input_ids = doc_input_ids + [sep_id]
    input_id_lst = [query_input_ids + doc_input_ids]
    token_type_ids_lst = [[0] * len(query_input_ids) + [1] * len(doc_input_ids)]
    position_ids_lst = [
        list(range(len(query_input_ids) + len(doc_input_ids)))]

    input_id_lst = pack_tensor_2D(input_id_lst, default=0, dtype=torch.int64)
    token_type_ids_lst = pack_tensor_2D(token_type_ids_lst, default=0, dtype=torch.int64)
    position_ids_lst = pack_tensor_2D(position_ids_lst, default=0,
                                      dtype=torch.int64)

    model.eval()
    with torch.no_grad():
        input_id_lst = input_id_lst.to(args.device)
        token_type_ids_lst = token_type_ids_lst.to(args.device)
        position_ids_lst = position_ids_lst.to(args.device)
        outputs = model(input_id_lst, token_type_ids_lst, position_ids_lst)
        scores = outputs.detach().cpu()
        norm_scores = torch.sigmoid(scores).numpy()
        norm_scores = np.squeeze(norm_scores, axis=-1)

    return norm_scores[0]


def randomized_evaluate_single_doc(args, model, queries, tokenizer, random_doc_dir, randomized_doc_file, qid):

    tokenized_file_path = random_doc_dir + '/tokenized/' + randomized_doc_file
    eval_dataset = MSMARCORandomDocTestDataset(qid, queries, tokenized_file_path, tokenizer,
                                               max_query_length=64, max_doc_length=445)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    collate_fn = get_collate_function_randomized_test()
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 num_workers=args.data_evaluate_num_workers, collate_fn=collate_fn)

    normalized_scores = []
    for batch in tqdm(eval_dataloader, desc='Evaluating'):
        model.eval()

        with torch.no_grad():
            batch = {k: v.to(args.device) for k, v in batch.items()}
            outputs = model(**batch)
            scores = outputs.detach().cpu()
            norm_scores_batch = torch.sigmoid(scores).numpy()
            norm_scores_batch = np.squeeze(norm_scores_batch, axis=-1)
            normalized_scores.extend(list(norm_scores_batch))

    return np.mean(normalized_scores)


def fRS_evaluate_and_save(args, model, tokenize_dir, tokenizer):

    qdr = read_qdr_pairs(args.bert_top100_ranked_list_path)
    evaluate_qid_num = 200
    qid_list = list(qdr.keys())[:evaluate_qid_num]

    # 保存结果
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    result_save_name = os.path.join(args.result_dir, 'cached_evaluate_all_f_RS_{}_{}_{}_{}_{}'.format(
        'test',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.task_name),
        str(args.similarity_threshold),
        str(args.perturbation_constraint)))

    if os.path.exists(result_save_name):
        predicted_qds = read_qds_pairs(result_save_name)
        for qid in predicted_qds:
            if qid in qid_list:
                if len(predicted_qds[qid]) == 100:
                    qid_list.remove(qid)
                else:
                    for did in predicted_qds[qid]:
                        qdr[qid].pop(did)
        output = open(result_save_name, 'a')
    else:

        output = open(result_save_name, 'w')

    queries = load_queries(tokenize_dir, mode='dev')

    for qid in tqdm(qid_list):
        ranked_docs = qdr[qid]

        test_doc_list = list(ranked_docs.keys())

        random_doc_dir = os.path.join(args.data_dir, 'random_test_docs' + str(args.similarity_threshold) + '_' + str(
            args.perturbation_constraint))

        if os.path.exists(random_doc_dir):
            files = os.listdir(random_doc_dir)

            for test_doc_id in tqdm(test_doc_list):
                file = str(test_doc_id) + '_num_' + str(args.num_random_sample)
                if file not in files:
                    print(file, 'not in the dir! please check!')
                else:
                    did = test_doc_id

                    estimated_test_doc_score = randomized_evaluate_single_doc(args, model, queries,
                                                                              tokenizer, random_doc_dir, file, qid)

                    write_single_doc_score_str = str(qid) + '\t' + str(did) + '\t' + str(estimated_test_doc_score) + '\n'
                    output.write(write_single_doc_score_str)



    output.close()


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="write whatever you want")
    parser.add_argument("--result_dir", default=None, type=str, required=True,
                        help="The output directory where the result will be written.")
    parser.add_argument("--num_random_sample", default=1000, type=int,
                        help="The number of random samples of each text.")
    parser.add_argument("--similarity_threshold", default=0.9, type=float,
                        help="The similarity constraint to be considered as synonym.")
    parser.add_argument("--perturbation_constraint", default=100, type=int,
                        help="The maximum size of perturbation set of each word")
    # Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--certify_K", default=20,
                        type=int, help="the top K document that you want to certify (cannot be ranked beyond it)")
    parser.add_argument("--bert_top100_ranked_list_path",
                        default='',
                        type=str, help="the top-100 ranked list produced by BERT")
    parser.add_argument("--query_tokenize_dir",
                        default='msmarco/document_ranking/tokenized_query_collection',
                        type=str, help="")
    parser.add_argument("--data_evaluate_num_workers",
                        default=20,
                        type=int, help="")
    parser.add_argument("--use_ori_k_score",
                        action='store_true',
                        help="use the document k's score of the original model instead of the randomized smoothing model")

    parser.add_argument("--collection_memmap_dir", type=str,
                        default="/msmarco/document_ranking/tokenized_collection_memmap")

    args = parser.parse_args()

    if os.path.exists(args.result_dir) and os.listdir(args.result_dir) and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.checkpoint_dir))

    # Setup CUDA, GPU

    os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("Process device: %s, n_gpu: %s",
                   device, args.n_gpu)

    # Set seed
    set_seed(args)

    # Prepare task
    args.task_name = args.task_name.lower()

    # Load pretrained model and tokenizer

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path)

    model = model_class.from_pretrained(args.model_name_or_path, config=config)

    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    model.to(args.device)

    fRS_evaluate_and_save(args, model, args.query_tokenize_dir, tokenizer)


if __name__ == "__main__":
    main()
