import os
import string
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
from nltk import word_tokenize
import networkx as nx
import argparse


def get_wordembd(embd_path):
    word_embd = {}
    embd_file = os.path.join(embd_path, 'counter-fitted-vectors.txt')
    with open(embd_file, "r") as f:
        tem = f.readlines()
        for line in tem:
            line = line.strip()
            line = line.split(' ')
            word = line[0]
            vec = line[1:len(line)]
            vec = [float(i) for i in vec]
            vec = np.asarray(vec)
            word_embd[word] = vec

    Name = embd_path + '/word_embd.pkl'
    output = open(Name, 'wb')
    pickle.dump(word_embd, output)
    output.close()


def get_vocabluary(dataset, data_path, embd_path, marco_path):
    print('Generate vocabulary')
    pkl_file = open(embd_path + '/word_embd.pkl', 'rb')
    word_embd = pickle.load(pkl_file)
    pkl_file.close()

    if dataset == 'msmarco-doc':
        vocab = {}
        doc_path = marco_path
        # ID \t url \t title\t content
        doc_content = []
        with open(doc_path, 'r') as df:
            i = 0
            for line in df:
                i += 1
                if i % 10000 == 0:
                    print('finished', i , 'lines')

                l = line.strip().split('\t')
                docid = ''
                if len(l) == 2:
                    docid = l[0]
                    title = ""
                    body = l[1]
                    url = ""
                elif len(l) == 3:
                    docid = l[0]
                    title = l[1]
                    body = l[2]
                    url = ""
                elif len(l) == 4:
                    docid = l[0]
                    url = l[1]
                    title = l[2]
                    body = l[3]

                if docid == '':
                    print('id is null', line)
                content = title + ' ' + body
                doc_content.append(content)
            for content in doc_content:
                if content:
                    tem_text = content.lower().translate(str.maketrans('', '', string.punctuation))
                    tem_text = tem_text.split(' ')
                    for word in tem_text:
                        if word in vocab.keys():
                            vocab[word]['freq'] = vocab[word]['freq'] + 1
                        else:
                            if word in word_embd.keys():
                                vocab[word] = {'vec': word_embd[word], 'freq': 1}

        Name = data_path + '/' + dataset + '_vocab.pkl'
        output = open(Name, 'wb')
        pickle.dump(vocab, output)
        output.close()
        print('Finish Generate msmarco-doc vocabulary')

    elif dataset == 'msmarco-pas':
        vocab = {}
        doc_path = marco_path
        # ID \t text \n
        doc_content = []
        with open(doc_path, 'r') as df:
            i = 0
            for line in df:
                i += 1
                if i % 10000 == 0:
                    print('finished', i, 'lines')
                l = line.strip().split('\t')
                docid = l[0]
                content = l[1]
                doc_content.append(content)
            for content in doc_content:
                if content:
                    tem_text = content.lower().translate(str.maketrans('', '', string.punctuation))
                    tem_text = tem_text.split(' ')
                    for word in tem_text:
                        if word in vocab.keys():
                            vocab[word]['freq'] = vocab[word]['freq'] + 1
                        else:
                            if word in word_embd.keys():
                                vocab[word] = {'vec': word_embd[word], 'freq': 1}

        Name = data_path + '/' + dataset + '_vocab.pkl'
        output = open(Name, 'wb')
        pickle.dump(vocab, output)
        output.close()
        print('Finish Generate msmarco-pas vocabulary')


def process_with_all_but_not_top(dataset, data_path):
    # code for processing word embd using all-but-not-top
    print('Process word embd using all-but-not-top')
    if dataset == 'msmarco-doc':
        pkl_file = open(data_path + '/msmarco-doc_vocab.pkl', 'rb')
    elif dataset == 'msmarco-pas':
        pkl_file = open(data_path + '/msmarco-pas_vocab.pkl', 'rb')

    vocab = pickle.load(pkl_file)
    pkl_file.close()

    num_word = len(vocab)

    assert 'high' in vocab
    dim_vec = len(vocab['high']['vec'])
    embd_matrix = np.zeros([num_word, dim_vec])
    embd_matrix0 = np.zeros([num_word, dim_vec])

    count = 0
    tem_list = []
    for key in vocab.keys():
        tem_vec = vocab[key]['vec']
        tem_vec = tem_vec / np.sqrt((tem_vec**2).sum())
        embd_matrix[count, :] = tem_vec
        tem_list.append(key)
        count += 1

    mean_embd_matrix = np.mean(embd_matrix, axis=0)
    for i in range(embd_matrix.shape[0]):
        embd_matrix0[i, :] = embd_matrix[i, :] - mean_embd_matrix
    covMat = np.cov(embd_matrix0, rowvar=0)
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    eigValIndice = np.argsort(-eigVals)
    eigValIndice = eigValIndice[0:8]
    n_eigVect=eigVects[:, eigValIndice]
    embd_matrix = embd_matrix0 - np.dot(np.dot(embd_matrix, n_eigVect), n_eigVect.T)

    if dataset == 'msmarco-doc':
        Name = data_path + '/msmarco-doc_embd_pca.pkl'
    elif dataset == 'msmarco-pas':
        Name = data_path + '/msmarco-pas_embd_pca.pkl'
    output = open(Name, 'wb')
    pickle.dump(embd_matrix, output)
    output.close()

    # update vocabulary
    count = 0
    for key in tem_list:
        vocab[key]['vec'] = embd_matrix[count, :].flatten()
        count += 1

    if dataset == 'msmarco-doc':
        Name = data_path + '/msmarco-doc_vocab_pca.pkl'
    elif dataset == 'msmarco-pas':
        Name = data_path + '/msmarco-pas_vocab_pca.pkl'

    output = open(Name, 'wb')
    pickle.dump(vocab, output)
    output.close()

    print('Finish Process word embd using all-but-not-top')


def get_word_substitution_table(dataset, data_path, similarity_threshold=0.5):
    print('Generate word substitude table')
    if dataset == 'msmarco-doc':
        pkl_file = open(data_path + '/msmarco-doc_vocab_pca.pkl', 'rb')
    elif dataset == 'msmarco-pas':
        pkl_file = open(data_path + '/msmarco-pas_vocab_pca.pkl', 'rb')
    vocab = pickle.load(pkl_file)
    pkl_file.close()

    counterfitted_neighbor = {}
    key_list = list(vocab.keys())
    similarity_num_threshold = 100000
    freq_threshold = 1
    neighbor_network_node_list = []
    neighbor_network_link_list = []

    num_word = len(key_list)
    dim_vec = vocab[key_list[0]]['vec'].shape[1]

    embd_matrix = np.zeros([num_word, dim_vec])
    for _ in range(len(key_list)):
        embd_matrix[_, :] = vocab[key_list[_]]['vec']

    for _ in tqdm(range(len(key_list))):
        word = key_list[_]

        if vocab[word]['freq'] > freq_threshold:

            counterfitted_neighbor[word] = []
            neighbor_network_node_list.append(word)

            dist_vec = np.dot(embd_matrix[_, :], embd_matrix.T)
            dist_vec = np.array(dist_vec).flatten()

            idxes = np.argsort(-dist_vec)
            idxes = np.where(dist_vec > similarity_threshold)
            idxes = idxes[0].tolist()

            tem_num_count = 0
            for ids in idxes:
                if key_list[ids] != word and vocab[key_list[ids]]['freq'] > freq_threshold:
                    counterfitted_neighbor[word].append(key_list[ids])
                    neighbor_network_link_list.append((word, key_list[ids]))
                    tem_num_count += 1
                    if tem_num_count >= similarity_num_threshold:
                        break

        if _ % 2000 == 0:
            neighbor = {'neighbor': counterfitted_neighbor, 'link': neighbor_network_link_list, 'node': neighbor_network_node_list}
            if dataset == 'msmarco-doc':
                Name = data_path + '/msmarco-doc_neighbor_constraint_pca' + str(similarity_threshold) + '.pkl'
            elif dataset == 'msmarco-pas':
                Name = data_path + '/msmarco-pas_neighbor_constraint_pca' + str(similarity_threshold) + '.pkl'
            output = open(Name, 'wb')
            pickle.dump(neighbor, output)
            output.close()

    neighbor = {'neighbor': counterfitted_neighbor, 'link': neighbor_network_link_list, 'node': neighbor_network_node_list}
    if dataset == 'msmarco-doc':
        Name = data_path + '/msmarco-doc_neighbor_constraint_pca' + str(similarity_threshold) + '.pkl'
    elif dataset == 'msmarco-pas':
        Name = data_path + '/msmarco-pas_neighbor_constraint_pca' + str(similarity_threshold) + '.pkl'
    output = open(Name, 'wb')
    pickle.dump(neighbor, output)
    output.close()
    print('Finish Generate word substitude table')


def get_perturbation_set(dataset, data_path, similarity_threshold=0.5, perturbation_constraint=100):

    # code for generate perturbation set
    print('Generate perturbation set')
    freq_threshold = 1

    if dataset == 'msmarco-doc':
        pkl_file = open(data_path + '/msmarco-doc_neighbor_constraint_pca' + str(similarity_threshold) + '.pkl', 'rb')
        neighbor = pickle.load(pkl_file)
        pkl_file.close()

        pkl_file = open(data_path + '/msmarco-doc_vocab_pca.pkl', 'rb')
        vocab = pickle.load(pkl_file)
        pkl_file.close()
    elif dataset == 'msmarco-pas':
        pkl_file = open(data_path + '/msmarco-pas_neighbor_constraint_pca' + str(similarity_threshold) + '.pkl', 'rb')
        neighbor = pickle.load(pkl_file)
        pkl_file.close()

        pkl_file = open(data_path + '/msmarco-pas_vocab_pca.pkl', 'rb')
        vocab = pickle.load(pkl_file)
        pkl_file.close()

    counterfitted_neighbor = neighbor['neighbor']
    neighbor_network_node_list = neighbor['node']
    neighbor_network_link_list = neighbor['link']
    perturb = {}

    size_threshold = perturbation_constraint

    key_list = list(vocab.keys())
    num_word = len(key_list)
    dim_vec = vocab[key_list[0]]['vec'].shape[1]
    embd_matrix = np.zeros([num_word, dim_vec])
    for _ in range(len(key_list)):
        embd_matrix[_, :] = vocab[key_list[_]]['vec']

    G = nx.Graph()
    for node in neighbor_network_node_list:
        G.add_node(node)
    for link in neighbor_network_link_list:
        G.add_edge(link[0], link[1])

    for c in nx.connected_components(G):
        nodeSet = G.subgraph(c).nodes()
        if len(nodeSet) > 1:
            if len(nodeSet) <= perturbation_constraint:
                tem_key_list = list(nodeSet)
                tem_num_word = len(tem_key_list)
                tem_embd_matrix = np.zeros([tem_num_word, dim_vec])

                for _ in range(len(tem_key_list)):
                    tem_embd_matrix[_, :] = vocab[tem_key_list[_]]['vec']

                for node in nodeSet:
                    perturb[node] = {'set': G.subgraph(c).neighbors(node), 'isdivide': 0}
                    dist_vec = np.dot(vocab[node]['vec'], tem_embd_matrix.T)
                    dist_vec = np.array(dist_vec).flatten()
                    idxes = np.argsort(-dist_vec)
                    tem_list = []
                    for ids in idxes:
                        if vocab[tem_key_list[ids]]['freq'] > freq_threshold:
                            tem_list.append(tem_key_list[ids])
                    perturb[node]['set'] = tem_list

            else:
                tem_key_list = list(nodeSet)
                tem_num_word = len(tem_key_list)
                tem_embd_matrix = np.zeros([tem_num_word, dim_vec])
                for _ in range(len(tem_key_list)):
                    tem_embd_matrix[_, :] = vocab[tem_key_list[_]]['vec']

                for node in tqdm(nodeSet):
                    perturb[node] = {'set': G.subgraph(c).neighbors(node), 'isdivide': 1}
                    if len(list(perturb[node]['set'])) > size_threshold:
                        raise ValueError('size_threshold is too small')

                    dist_vec = np.dot(vocab[node]['vec'], tem_embd_matrix.T)
                    dist_vec = np.array(dist_vec).flatten()
                    idxes = np.argsort(-dist_vec)
                    tem_list = []
                    tem_count = 0
                    for ids in idxes:
                        if vocab[tem_key_list[ids]]['freq'] > freq_threshold:
                            tem_list.append(tem_key_list[ids])
                            tem_count += 1
                        if tem_count == size_threshold:
                            break
                    perturb[node]['set'] = tem_list

    if dataset == 'msmarco-doc':
        Name = data_path + '/msmarco-doc_perturbation_constraint_pca' + str(similarity_threshold) + '_' + str(size_threshold) + '.pkl'
    elif dataset == 'msmarco-pas':
        Name = data_path + '/msmarco-pas_perturbation_constraint_pca' + str(similarity_threshold) + '_' + str(size_threshold) + '.pkl'
    output = open(Name, 'wb')
    pickle.dump(perturb, output)
    output.close()
    print('generate perturbation set finishes')
    print('-'*89)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--dataset", default='msmarco-doc', type=str, required=True,
                        help="The name of data set: msmarco-pas or msmarco-doc")
    parser.add_argument("--data_path", default='', type=str, required=True,
                        help="The input data dir.")
    parser.add_argument("--marco_path", default=''
                        , type=str, required=True, help=".")
    parser.add_argument("--embd_path", default='', type=str, required=True,
                        help="The data dir of embedding table.")
    parser.add_argument("--similarity_threshold", default=0.5, type=float,
                        help="The similarity constraint to be considered as synonym.")
    parser.add_argument("--perturbation_constraint", default=100, type=int,
                        help="The maximum size of perturbation set of each word")

    args = parser.parse_args()
    print(args)

    data_path = args.data_path
    embd_path = args.embd_path
    marco_path = args.marco_path
    dataset = args.dataset
    similarity_threshold = args.similarity_threshold
    perturbation_constraint = args.perturbation_constraint

    if dataset == 'msmarco-pas':
        dataset_name = 'msmarco-pas'
    elif dataset == 'msmarco-doc':
        dataset_name = 'msmarco-doc'
    else:
        raise ValueError('dataset not valid. Choose from msmarco-pas or msmarco-doc')

    embd_file = embd_path + '/word_embd.pkl'

    if not os.path.exists(embd_file):
        get_wordembd(embd_path)

    if not os.path.exists(data_path + '/' + dataset_name + '_vocab.pkl'):
        get_vocabluary(dataset, data_path, embd_path, marco_path)

    if not os.path.exists(data_path + '/' + dataset_name + '_embd_pca.pkl') or not os.path.exists(data_path + '/' + dataset_name + '_vocab_pca.pkl'):
        process_with_all_but_not_top(dataset, data_path)

    if not os.path.exists(data_path + '/' + dataset_name + '_neighbor_constraint_pca' + str(similarity_threshold) + '.pkl'):
        get_word_substitution_table(dataset, data_path, similarity_threshold=similarity_threshold)

    if not os.path.exists(data_path + '/' + dataset_name + '_perturbation_constraint_pca' + str(similarity_threshold) + '_' + str(perturbation_constraint) + '.pkl'):
        get_perturbation_set(dataset, data_path, similarity_threshold=similarity_threshold, perturbation_constraint=perturbation_constraint)

    if dataset == 'msmarco-doc':
        pkl_file = open(data_path + '/msmarco-doc_perturbation_constraint_pca' + str(similarity_threshold) + '_' + str(perturbation_constraint) + '.pkl', 'rb')
        neighbor = pickle.load(pkl_file)
        print(neighbor)
        pkl_file.close()

    if dataset == 'msmarco-pas':
        pkl_file = open(data_path + '/msmarco-pas_perturbation_constraint_pca' + str(similarity_threshold) + '_' + str(perturbation_constraint) + '.pkl', 'rb')
        neighbor = pickle.load(pkl_file)
        print(neighbor)
        pkl_file.close()


if __name__ == "__main__":
    main()



















