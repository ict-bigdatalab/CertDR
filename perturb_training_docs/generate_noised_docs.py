from tqdm import tqdm
import pickle, argparse
from transformers import BertTokenizer

from word_sub_helper import WordSubstitude
from marcodoc.dataset import CollectionDataset


def generate_ori_docs(dataset_prefix, bert_tokenizer_path, collection_memmap_dir):

    collection = CollectionDataset(collection_memmap_dir)

    bert_tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_path)
    # random smoother

    i = 0
    ori_doc_list = []
    for did in tqdm(collection.pids):
        ori_token_ids = collection[did]
        ori_token_ids = ori_token_ids[:445]
        ori_doc_text = bert_tokenizer.decode(ori_token_ids)
        ori_doc_list.append(ori_doc_text)

    read_path = dataset_prefix + 'ori_doc_445'
    with open(read_path, 'w') as wf:
        for line in ori_doc_list:
            wf.write(line + '\n')


def generate_perturbed_docs(similarity_threshold, perturbation_constraint,
                            dataset_prefix, dataset_name):

    pkl_file = open(dataset_prefix + dataset_name + '_perturbation_constraint_pca' + str(similarity_threshold) + '_' + str(
            perturbation_constraint) + '.pkl', 'rb')
    word_substituide_table = pickle.load(pkl_file)
    random_smooth = WordSubstitude(word_substituide_table)

    ori_doc_list = []

    read_path = dataset_prefix + 'ori_doc_445'
    with open(read_path, 'r') as rf:
        for line in rf:
            ss = line.strip()
            ori_doc_list.append(ss)

    import numpy as np
    i = 0
    perturbed_doc_list = []
    for ori_doc in tqdm(ori_doc_list):
        perturbed_doc = str(random_smooth.get_perturbed_batch(np.array([[ori_doc]]))[0][0])
        perturbed_doc_list.append(perturbed_doc)

    write_path = dataset_prefix + 'perturbed_doc_445_' + str(perturbation_constraint)
    wf = open(write_path, 'w')
    for pd in tqdm(perturbed_doc_list):
        wf.write(pd + '\n')


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--similarity_threshold", default=0.9, type=float,
                        help="The similarity constraint to be considered as synonym.")
    parser.add_argument("--perturbation_constraint", default=100, type=int,
                        help="The maximum size of perturbation set of each word")
    parser.add_argument("--dataset_prefix", default='/data/msmarco-doc/', type=str,
                        help="Your dataset prefix")
    parser.add_argument("--dataset_name", default='msmarco-doc', type=str,
                        help="Your dataset name")
    parser.add_argument("--bert_tokenizer_path", type=str, default="./data/tokenize")
    parser.add_argument("--collection_memmap_dir", type=str, default="/msmarco/document_ranking/tokenized_collection_memmap")

    args = parser.parse_args()

    generate_ori_docs(args.dataset_prefix, args.bert_tokenizer_path, args.collection_memmap_dir)
    generate_perturbed_docs(args.similarity_threshold, args.perturbation_constraint,
                            args.dataset_prefix, args.dataset_name)


if __name__ == "__main__":
    main()