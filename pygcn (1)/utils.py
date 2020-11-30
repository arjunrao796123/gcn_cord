import numpy as np
import scipy.sparse as sp
import torch
import random
from sklearn.metrics import classification_report
from scipy import sparse
folderName = r"/home/arjun/Downloads/cord_text-20201130T203217Z-001/cord_text/"
import os
import nltk
import gensim


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora",adj='', features='', labels='',index = ''):
    """Load citation network dataset (cora only for now)"""
    '''    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    '''
    # features
    #print(idx_features_labels[:, 1:-1])
    #features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    #labels = encode_onehot(idx_features_labels[:, -1])
    #labels = labels.iloc[:,13:]
    labels= labels.iloc[:, 13:]
    print(labels.shape)
    print(labels.columns)

    # build graph
    '''
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    '''
    #adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
    #                    shape=(labels.shape[0], labels.shape[0]),
    #                   dtype=np.float32)


    # build symmetric adjacency matrix
    #features = normalize(features)
    #adj = sparse.csr_matrix(adj)
    #adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    #adj = normalize(adj + sp.eye(adj.shape[0]))
    idx_train = range(index[int(0.6*len(index))])
    idx_val = range(index[int(0.6*len(index))],index[int(0.8*len(index))])
    idx_test = range(index[int(0.8*len(index))],index[int(1*len(index))-1])

    #idx_train = range(index[1])
    #idx_val = range(index[1])
    #idx_test = range(index[1])

    #idx_train = train['A'].astype(int).values
    #print(idx_train)
    #idx_val = validate['A'].astype(int).values
    #idx_test = test['A'].astype(int).values
    print(len(idx_train))
    print(len(idx_val))
    print(len(idx_test))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    #adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels,flag=True,target_names = ''):

    #print(output.shape)
    #print(output)
    preds = output.max(1)[1].type_as(labels)
    #print(labels,'labels')
    print(labels,'output')
    print(preds,'preds2')
    if flag is False:
        print(target_names)
        print(classification_report(labels.detach().cpu(),preds.detach().cpu(),labels=range(len(target_names)),target_names = target_names))
    correct = preds.eq(labels).double()

    correct = correct.sum()

    #print(correct / len(labels),'correct / len(labels)')
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
