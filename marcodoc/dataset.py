import os
import math
import json
import torch
import logging
import pickle

import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
from torch.utils.data import Dataset
logger = logging.getLogger(__name__)


class CollectionDataset:
    def __init__(self, collection_memmap_dir):
        self.pids = np.memmap(f"{collection_memmap_dir}/pids.memmap", dtype='int32',)
        self.lengths = np.memmap(f"{collection_memmap_dir}/lengths.memmap", dtype='int32',)
        self.collection_size = len(self.pids)
        self.token_ids = np.memmap(f"{collection_memmap_dir}/token_ids.memmap", 
                dtype='int32', shape=(self.collection_size, 512))
    
    def __len__(self):
        return self.collection_size

    def __getitem__(self, item):
        index = np.argwhere(self.pids == item)[0][0]

        return self.token_ids[index, :self.lengths[index]].tolist()


def load_queries(tokenize_dir, mode):
    queries = dict()
    for line in tqdm(open(f"{tokenize_dir}/msmarco-doc{mode}-queries.tokenized.json"), desc="queries"):
        data = json.loads(line)
        queries[int(data['id'])] = data['ids']
    return queries


def load_randomized_docs(randomized_doc_file_path):
    rdm_doc_file = open(randomized_doc_file_path, 'rb')
    randomized_doc_list = pickle.load(rdm_doc_file)
    return randomized_doc_list


def load_querydoc_pairs(msmarco_dir, mode):
    qids, pids, labels = [], [], []
    if mode == "train":
        for line in tqdm(open(f"{msmarco_dir}/train_triples_ids_10neg"),
                desc="load train triples"):
            qid, pos_pid, neg_pid = line.split("\t")
            qid, pos_pid, neg_pid = int(qid), int(pos_pid[1:]), int(neg_pid[1:])
            qids.append(qid)
            pids.append(pos_pid)
            labels.append(1)
            qids.append(qid)
            pids.append(neg_pid)
            labels.append(0)
    else:
        for line in open(f"{msmarco_dir}/msmarco-doc{mode}-top100.tsv"):  ### mode = dev
            qid, _, pid, _, _, _ = line.split(" ")
            qids.append(int(qid))
            pids.append(int(pid[1:]))
    if not mode == "train":
        labels, qrels = None, None
    return qids, pids, labels   #, qrels'


def load_querydoc_pairs_200_query(msmarco_dir, mode):
    qids, pids, labels = [], [], []
    if mode == "train":
        pass
    else:
        for line in open(f"{msmarco_dir}/msmarco-doc{mode}-top100.tsv"):
            qid, _, pid, _, _, _ = line.split(" ")
            qids.append(int(qid))
            pids.append(int(pid[1:]))
        qids = qids[:(200 * 100)]
        pids = pids[:(200 * 100)]

    if not mode == "train":
        labels, qrels = None, None
    return qids, pids, labels   #, qrel


def load_querydoc_pairs_noisy(msmarco_dir, mode, small_test):
    qids, pids, labels = [], [], []
    if mode == "train":
        for line in tqdm(open(f"{msmarco_dir}/train_triples_ids_10neg"),
                desc="load train triples"):
            qid, pos_pid, neg_pid = line.split("\t")
            qid, pos_pid, neg_pid = int(qid), int(pos_pid[1:]), int(neg_pid[1:])
            qids.append(qid)
            pids.append(pos_pid)
            labels.append(1)
            qids.append(qid)
            pids.append(neg_pid)
            labels.append(0)
    else:
        if small_test:
            TEST_PATH = msmarco_dir + '/msmarco-docdev-top100_519.tsv'
        else:
            TEST_PATH = msmarco_dir + '/msmarco-docdev-top100.tsv'
        for line in open(TEST_PATH):  # mode = dev
            qid, _, pid, _, _, _ = line.split(" ")
            qids.append(int(qid))
            pids.append(int(pid[1:]))
    if not mode == "train":
        labels, qrels = None, None
    return qids, pids, labels   #, qrels


def load_querydoc_pairs_attacked_docs(doc_path, mode, bert_tokenizer_path):
    qids, pids, labels = [], [], []
    doc_token_dict = {}
    tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_path)

    for line in open(doc_path):  # mode = dev

        if len(line.strip().split('\t')) == 3:
            qidpluspid, doc_content, _ = line.strip().split('\t')
        elif len(line.strip().split('\t')) == 2:
            qidpluspid, doc_content= line.strip().split('\t')
        qid, pid = qidpluspid.split('_')
        doc_tokens = tokenizer.encode(doc_content, add_special_tokens=False)
        qids.append(int(qid))
        pids.append(int(pid))
        doc_token_dict[qidpluspid] = doc_tokens
    return qids, pids, doc_token_dict


class MSMARCODataset(Dataset):
    def __init__(self, mode, msmarco_dir,
            collection_memmap_dir, tokenize_dir, tokenizer_dir,
            max_query_length=64, max_doc_length=445):

        self.collection = CollectionDataset(collection_memmap_dir)
        self.queries = load_queries(tokenize_dir, mode)
        self.qids, self.pids, self.labels = load_querydoc_pairs(msmarco_dir, mode)
        self.mode = mode
        tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, item):
        qid, pid = self.qids[item], self.pids[item]
        query_input_ids, doc_input_ids = self.queries[qid], self.collection[pid]
        query_input_ids = query_input_ids[:self.max_query_length]
        query_input_ids = [self.cls_id] + query_input_ids + [self.sep_id]
        doc_input_ids = doc_input_ids[:self.max_doc_length]
        doc_input_ids = doc_input_ids + [self.sep_id]

        ret_val = {
            "query_input_ids": query_input_ids,
            "doc_input_ids": doc_input_ids,
            "qid": qid,
            "docid" : pid
        }
        if self.mode == "train":
            label = self.labels[item]
            ret_val["label"] = label
        return ret_val


class MSMARCO_200QUERY_Dataset(Dataset):
    def __init__(self, mode, msmarco_dir,
            collection_memmap_dir, tokenize_dir, tokenizer_dir,
            max_query_length=64, max_doc_length=445):

        self.collection = CollectionDataset(collection_memmap_dir)
        self.queries = load_queries(tokenize_dir, mode)
        self.qids, self.pids, self.labels = load_querydoc_pairs_200_query(msmarco_dir, mode)
        self.mode = mode
        tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, item):
        qid, pid = self.qids[item], self.pids[item]
        query_input_ids, doc_input_ids = self.queries[qid], self.collection[pid]
        query_input_ids = query_input_ids[:self.max_query_length]
        query_input_ids = [self.cls_id] + query_input_ids + [self.sep_id]
        doc_input_ids = doc_input_ids[:self.max_doc_length]
        doc_input_ids = doc_input_ids + [self.sep_id]

        ret_val = {
            "query_input_ids": query_input_ids,
            "doc_input_ids": doc_input_ids,
            "qid": qid,
            "docid" : pid
        }
        if self.mode == "train":
            label = self.labels[item]
            ret_val["label"] = label
        return ret_val


class MSMARCONoisyDocDataset(Dataset):
    def __init__(self, mode, msmarco_dir,
            noisy_collection_memmap_dir, tokenize_dir, tokenizer_dir,
            max_query_length=64, max_doc_length=445, small_test=False):

        self.collection = CollectionDataset(noisy_collection_memmap_dir)
        self.queries = load_queries(tokenize_dir, mode)
        self.qids, self.pids, self.labels = load_querydoc_pairs_noisy(msmarco_dir, mode, small_test)
        self.mode = mode
        tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, item):
        qid, pid = self.qids[item], self.pids[item]
        query_input_ids, doc_input_ids = self.queries[qid], self.collection[pid]
        query_input_ids = query_input_ids[:self.max_query_length]
        query_input_ids = [self.cls_id] + query_input_ids + [self.sep_id]
        doc_input_ids = doc_input_ids[:self.max_doc_length]
        doc_input_ids = doc_input_ids + [self.sep_id]

        ret_val = {
            "query_input_ids": query_input_ids,
            "doc_input_ids": doc_input_ids,
            "qid": qid,
            "docid" : pid
        }
        if self.mode == "train":
            label = self.labels[item]
            ret_val["label"] = label
        return ret_val


class MSMARCODADocDataset(Dataset):
    def __init__(self, mode, msmarco_dir,
                 ori_collection_memmap_dir, aug1_collection_memmap_dir, aug2_collection_memmap_dir,
                 tokenize_dir, tokenizer_dir, max_query_length=64, max_doc_length=445, small_test=True):
        self.ori_collection = CollectionDataset(ori_collection_memmap_dir)
        self.aug1_collection = CollectionDataset(aug1_collection_memmap_dir)
        self.aug2_collection = CollectionDataset(aug2_collection_memmap_dir)
        self.queries = load_queries(tokenize_dir, mode)
        self.qids, self.pids, self.labels = load_querydoc_pairs_noisy(msmarco_dir, mode, small_test)
        self.mode = mode
        tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, item):
        qid, pid = self.qids[item], self.pids[item]
        query_input_ids = self.queries[qid]
        ori_doc_input_ids = self.ori_collection[pid]
        aug1_doc_input_ids = self.aug1_collection[pid]
        aug2_doc_input_ids = self.aug2_collection[pid]

        query_input_ids = query_input_ids[:self.max_query_length]
        query_input_ids = [self.cls_id] + query_input_ids + [self.sep_id]

        ori_doc_input_ids = ori_doc_input_ids[:self.max_doc_length]
        ori_doc_input_ids = ori_doc_input_ids + [self.sep_id]

        aug1_doc_input_ids = aug1_doc_input_ids[:self.max_doc_length]
        aug1_doc_input_ids = aug1_doc_input_ids + [self.sep_id]

        aug2_doc_input_ids = aug2_doc_input_ids[:self.max_doc_length]
        aug2_doc_input_ids = aug2_doc_input_ids + [self.sep_id]

        ret_val = {
            "query_input_ids": query_input_ids,
            "ori_doc_input_ids": ori_doc_input_ids,
            "aug1_doc_input_ids": aug1_doc_input_ids,
            "aug2_doc_input_ids": aug2_doc_input_ids,
            "qid": qid,
            "docid" : pid
        }
        if self.mode == "train":
            label = self.labels[item]
            ret_val["label"] = label
        return ret_val


class MSMARCORandomDocTestDataset(Dataset):
    def __init__(self, query_id, queries, randomized_doc_file_path, tokenizer,
            max_query_length=64, max_doc_length=445, small_test=False):

        self.randomized_doc_list = load_randomized_docs(randomized_doc_file_path)
        mode = 'dev'
        self.queries = queries
        self.qid = query_id
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length

    def __len__(self):
        return len(self.randomized_doc_list)

    def __getitem__(self, item):

        qid = self.qid

        query_input_ids, doc_input_ids = self.queries[qid], self.randomized_doc_list[item]
        query_input_ids = query_input_ids[:self.max_query_length]
        query_input_ids = [self.cls_id] + query_input_ids + [self.sep_id]
        doc_input_ids = doc_input_ids[:self.max_doc_length]
        doc_input_ids = doc_input_ids + [self.sep_id]

        ret_val = {
            "query_input_ids": query_input_ids,
            "doc_input_ids": doc_input_ids
        }

        return ret_val


class MSMARCODataset_test_attacked_docs(Dataset):
    def __init__(self, mode, msmarco_dir, tokenize_dir, tokenizer_dir,
            max_query_length=64, max_doc_length=445):
        self.queries = load_queries(tokenize_dir, mode)
        self.qids, self.pids, self.doc_token_dict = load_querydoc_pairs_attacked_docs(msmarco_dir, mode, tokenizer_dir)
        self.mode = mode
        tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, item):

        qid, pid = self.qids[item], self.pids[item]
        qid_plus_pid = str(qid) + '_' + str(pid)
        query_input_ids, doc_input_ids = self.queries[qid], self.doc_token_dict[qid_plus_pid]
        query_input_ids = query_input_ids[:self.max_query_length]
        query_input_ids = [self.cls_id] + query_input_ids + [self.sep_id]
        doc_input_ids = doc_input_ids[:self.max_doc_length]
        doc_input_ids = doc_input_ids + [self.sep_id]

        ret_val = {
            "query_input_ids": query_input_ids,
            "doc_input_ids": doc_input_ids,
            "qid": qid,
            "docid" : pid
        }
        if self.mode == "train":
            label = self.labels[item]
            ret_val["label"] = label
        return ret_val


def pack_tensor_2D(lstlst, default, dtype, length=None):
    batch_size = len(lstlst)
    length = length if length is not None else max(len(l) for l in lstlst)
    tensor = default * torch.ones((batch_size, length), dtype=dtype)
    for i, l in enumerate(lstlst):
        tensor[i, :len(l)] = torch.tensor(l, dtype=dtype)
    return tensor


def get_collate_function(mode):
    def collate_function(batch):
        input_ids_lst = [x["query_input_ids"] + x["doc_input_ids"] for x in batch]
        token_type_ids_lst = [[0]*len(x["query_input_ids"]) + [1]*len(x["doc_input_ids"]) 
            for x in batch]
        position_ids_lst = [list(range(len(x["query_input_ids"]) + len(x["doc_input_ids"]))) for x in batch]
        data = {
            "input_ids": pack_tensor_2D(input_ids_lst, default=0, dtype=torch.int64),
            "token_type_ids": pack_tensor_2D(token_type_ids_lst, default=0, dtype=torch.int64),
            "position_ids": pack_tensor_2D(position_ids_lst, default=0, dtype=torch.int64),
        }
        qid_lst = [x['qid'] for x in batch]
        docid_lst = [x['docid'] for x in batch]
        if mode == "train":
            data["labels"] = torch.tensor([x["label"] for x in batch], dtype=torch.int64)
        return data, qid_lst, docid_lst
    return collate_function


def get_collate_function_randomized_test():
    def collate_function(batch):
        input_ids_lst = [x["query_input_ids"] + x["doc_input_ids"] for x in batch]
        token_type_ids_lst = [[0]*len(x["query_input_ids"]) + [1]*len(x["doc_input_ids"])
            for x in batch]
        position_ids_lst = [list(range(len(x["query_input_ids"]) + len(x["doc_input_ids"]))) for x in batch]
        data = {
            "input_ids": pack_tensor_2D(input_ids_lst, default=0, dtype=torch.int64),
            "token_type_ids": pack_tensor_2D(token_type_ids_lst, default=0, dtype=torch.int64),
            "position_ids": pack_tensor_2D(position_ids_lst, default=0, dtype=torch.int64),
        }

        return data
    return collate_function


def get_collate_function_for_DA(mode):
    def collate_function(batch):
        input_ids_lst = []
        token_type_ids_lst = []
        position_ids_lst = []
        qid_lst = []
        docid_lst = []

        for x in batch:
            input_ids_lst.append(x["query_input_ids"] + x["ori_doc_input_ids"])
            input_ids_lst.append(x["query_input_ids"] + x["aug1_doc_input_ids"])
            input_ids_lst.append(x["query_input_ids"] + x["aug2_doc_input_ids"])

            token_type_ids_lst.append([0]*len(x["query_input_ids"]) + [1]*len(x["ori_doc_input_ids"]))
            token_type_ids_lst.append([0] * len(x["query_input_ids"]) + [1] * len(x["aug1_doc_input_ids"]))
            token_type_ids_lst.append([0] * len(x["query_input_ids"]) + [1] * len(x["aug2_doc_input_ids"]))

            position_ids_lst.append(list(range(len(x["query_input_ids"]) + len(x["ori_doc_input_ids"]))))
            position_ids_lst.append(list(range(len(x["query_input_ids"]) + len(x["aug1_doc_input_ids"]))))
            position_ids_lst.append(list(range(len(x["query_input_ids"]) + len(x["aug2_doc_input_ids"]))))

            qid_lst.append(x['qid'])
            qid_lst.append(x['qid'])
            qid_lst.append(x['qid'])

            docid_lst.append(x['docid'])
            docid_lst.append(x['docid'])
            docid_lst.append(x['docid'])

        data = {
            "input_ids": pack_tensor_2D(input_ids_lst, default=0, dtype=torch.int64),
            "token_type_ids": pack_tensor_2D(token_type_ids_lst, default=0, dtype=torch.int64),
            "position_ids": pack_tensor_2D(position_ids_lst, default=0, dtype=torch.int64),
        }

        if mode == "train":
            label_lst = []
            for x in batch:
                label_lst.append(x["label"])
                label_lst.append(x["label"])
                label_lst.append(x["label"])
            data["labels"] = torch.tensor(label_lst, dtype=torch.int64)
        return data, qid_lst, docid_lst
    return collate_function


def _test_dataset():
    dataset = MSMARCODataset(mode="train")
    for data in dataset:
        tokens = dataset.tokenizer.convert_ids_to_tokens(data["query_input_ids"])
        print(tokens)
        tokens = dataset.tokenizer.convert_ids_to_tokens(data["doc_input_ids"])
        print(tokens)
        print(data['qid'], data['docid'], data['rel_docs'])
        print()
        k = input()
        if k == "q":
            break
