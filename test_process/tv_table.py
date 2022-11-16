import os, pickle
import string

from tqdm import tqdm
import numpy as np


class Calculate_tv:
    def __init__(self, table_tv):
        self.table = table_tv
        self.key_dict = set(table_tv.keys())
        self.exclude = set(string.punctuation)

    def get_tv(self, text):
        # input a text (string)
        tem_text = text.split(' ')
        tv_list = np.zeros(len(tem_text))
        if tem_text[0]:
            for j in range(len(tem_text)):
                if tem_text[j][-1] in self.exclude:
                    tem_text[j] = tem_text[j][0:-1]
                if tem_text[j] in self.key_dict:
                    tv_list[j] = self.table[tem_text[j]]
                else:
                    tv_list[j] = 1.
        return np.sort(tv_list)


def calculate_tv_table(args, perturb):
    similarity_threshold = args.similarity_threshold
    data_dir = args.data_dir

    dataset_name = args.task_name

    # reading vocabulary
    pkl_file = open(os.path.join(data_dir, dataset_name + '_vocab_pca.pkl'), 'rb')
    data_vocab = pickle.load(pkl_file)
    pkl_file.close()

    # reading neighbor set
    pkl_file = open(
        os.path.join(data_dir, dataset_name + '_neighbor_constraint_pca' + str(similarity_threshold) + '.pkl'), 'rb')
    data_neighbor = pickle.load(pkl_file)
    pkl_file.close()

    data_neighbor = data_neighbor['neighbor']

    total_intersect = 0
    total_freq = 0

    counterfitted_tv = {}
    for key in tqdm(data_neighbor.keys()):
        if not key in perturb.keys():
            counterfitted_tv[key] = 1

            total_intersect += data_vocab[key]['freq'] * 1
            total_freq += data_vocab[key]['freq']

        elif perturb[key]['isdivide'] == 0:
            counterfitted_tv[key] = 1

            total_intersect += data_vocab[key]['freq'] * 1
            total_freq += data_vocab[key]['freq']

        else:
            key_neighbor = data_neighbor[key]
            cur_min = 10.
            num_perb = len(perturb[key]['set'])
            for neighbor in key_neighbor:
                num_neighbor_perb = len(perturb[neighbor]['set'])
                num_inter_perb = len(list(set(perturb[neighbor]['set']).intersection(set(perturb[key]['set']))))
                tem_min = num_inter_perb / num_perb
                if tem_min < cur_min:
                    cur_min = tem_min
            counterfitted_tv[key] = cur_min

            total_intersect += data_vocab[key]['freq'] * cur_min
            total_freq += data_vocab[key]['freq']

    Name = os.path.join(data_dir, dataset_name + '_counterfitted_tv_pca' + str(similarity_threshold) + '_' + str(
        args.perturbation_constraint) + '.pkl')
    output = open(Name, 'wb')
    pickle.dump(counterfitted_tv, output)
    output.close()
    print('calculate total variation finishes')
    print('-' * 89)

    return counterfitted_tv