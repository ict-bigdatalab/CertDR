import os, random, pickle, sys
from os.path import abspath, join, dirname
sys.path.insert(0, join(abspath(dirname(__file__)), '..'))
import os, gc
import json
import argparse
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
from multiprocessing import Pool


tokenizer = BertTokenizer.from_pretrained("/bert_pretrained_model/bert-base-uncased")


def tokenize_doc_content(input_args):
    i = 0
    doc_list = input_args
    res_lines = []
    for doc in doc_list:
        text = doc['content']
        doc_id = doc['docid']
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        res_json = json.dumps(
            {"id": doc_id, "ids": ids}
        )
        res_lines.append(res_json)
        i += 1
        if i%10000 == 0:
            print('finished', i, 'lines')
    print('finished a subprocess!')
    del doc_list, input_args
    gc.collect()

    return res_lines


def tokenize_MSMARCOdocument_ranking_collection(input_file, collection_memmap_dir):

    from marcodoc.dataset import CollectionDataset
    collection = CollectionDataset(collection_memmap_dir)
    total_size = sum(1 for _ in open(input_file))
    docs_list = []

    index = 0
    # read all the docs
    for line in tqdm(open(input_file), total=total_size,
            desc=f"Read contents: {os.path.basename(input_file)}"):
        text = line.strip()
        doc_id = collection.pids[index]
        index += 1

        doc_dict = {}
        doc_dict['docid'] = int(doc_id)
        doc_dict['content'] = text
        docs_list.append(doc_dict)
    new_docs_list = docs_list

    # seperate the data
    docs_list_seps = []
    docs_list_s = []
    for j in range(len(new_docs_list)):
        if (j != 0) and (j % 20000 == 0):
            docs_list_seps.append(docs_list_s)
            docs_list_s = []
            print("finished given", j, "docs")
        docs_list_s.append(new_docs_list[j])
        if j == (len(new_docs_list) - 1):
            docs_list_seps.append(docs_list_s)
            docs_list_s = []

    del docs_list, new_docs_list
    gc.collect()

    # multi-process the data
    arg_list = [docs_list_ss for docs_list_ss in docs_list_seps]

    pool = Pool(50)
    res = pool.map(tokenize_doc_content, arg_list)

    del arg_list
    gc.collect()

    # write to the file
    output_file = args.output_dir + '/tokenized_perturb_collection_' + str(args.similarity_threshold)
    write_index = 0
    outFile = open(output_file, 'w')
    for doc_list in res:
        for doc_json in doc_list:
            outFile.write(doc_json)
            outFile.write("\n")
            write_index += 1
            if write_index % 30000 == 0:
                print('finished writing', write_index, 'lines')


def tokenize_MSMARCOpassage_ranking_collection(input_file, collection_memmap_dir):
    from marcopas.dataset import CollectionDataset
    collection = CollectionDataset(collection_memmap_dir)
    total_size = sum(1 for _ in open(input_file))
    docs_list = []

    index = 0
    # read all the docs
    for line in tqdm(open(input_file), total=total_size,
            desc=f"Read contents: {os.path.basename(input_file)}"):
        text = line.strip()
        doc_id = collection.pids[index]
        index += 1
        doc_dict = {}
        doc_dict['docid'] = int(doc_id)
        doc_dict['content'] = text
        docs_list.append(doc_dict)

    new_docs_list = docs_list

    # seperate the data
    docs_list_seps = []
    docs_list_s = []
    for j in range(len(new_docs_list)):
        if (j != 0) and (j % 40000 == 0):
            docs_list_seps.append(docs_list_s)
            docs_list_s = []
            print("finished given", j, "docs")
        docs_list_s.append(new_docs_list[j])
        if j == (len(new_docs_list) - 1):
            docs_list_seps.append(docs_list_s)
            docs_list_s = []

    del docs_list, new_docs_list
    gc.collect()

    # multi-process the data
    arg_list = [docs_list_ss for docs_list_ss in docs_list_seps]

    pool = Pool(50)
    res = pool.map(tokenize_doc_content, arg_list)

    del arg_list
    gc.collect()

    # write to the file
    output_file = args.output_dir + '/tokenized_perturb_collection_' + str(args.similarity_threshold)
    write_index = 0
    outFile = open(output_file, 'w')
    for doc_list in res:
        for doc_json in doc_list:
            outFile.write(doc_json)
            outFile.write("\n")
            write_index += 1
            if write_index % 30000 == 0:
                print('finished writing', write_index, 'lines')


def tokenize_collection_document_ranking(args):
    collection_input = f"{args.msmarco_dir}/perturbed_doc_445_" + str(args.similarity_threshold)
    tokenize_MSMARCOdocument_ranking_collection(collection_input, args.collection_memmap_dir)


def tokenize_collection_passage_ranking(args):
    collection_input = f"{args.msmarco_dir}/perturbed_docs_" + str(args.similarity_threshold)
    tokenize_MSMARCOpassage_ranking_collection(collection_input, args.collection_memmap_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--msmarco_dir", type=str, default="/data/users/wuchen/pythonprojects/prada_certify/data/msmarco-doc")
    parser.add_argument("--output_dir", type=str, default="/data/users/wuchen/pythonprojects/prada_certify/data/msmarco-doc/collection_memmap")
    parser.add_argument("--dataset_type", type=str, default="document_ranking")
    parser.add_argument("--collection_memmap_dir", type=str, default="/msmarco/document_ranking/tokenized_collection_memmap")
    parser.add_argument("--similarity_threshold", default=0.9, type=float,
                        help="The similarity constraint to be considered as synonym.")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.dataset_type == 'document_ranking':
        tokenize_collection_document_ranking(args)

    if args.dataset_type == 'passage_ranking':
        tokenize_collection_passage_ranking(args)
