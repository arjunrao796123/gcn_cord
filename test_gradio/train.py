from __future__ import division
from __future__ import print_function
from scipy import sparse
import scipy.sparse as sp
import torch
import json
from scipy import sparse
import time
import argparse
import numpy as np
import cv2
#from grapher import ObjectTree, Graph
import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from utils import load_data, accuracy
from models import GCN
import os

loss_plot = []

import warnings
warnings.filterwarnings('ignore')

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=True,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=69, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=5e-3,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data for 3 images
'''
folderName_json = r"/home/arjun/receipts/json/test/"
folderName_image = r'/home/arjun/Graph-Convolution-on-Structured-Documents/data/test/'
folderNameReceipt_csv = '/home/arjun/receipts_by_cord/test/'
dataframe_collection = {}
Image_count = 0
list_folderName_json = list(os.listdir(folderName_json))
list_folderName_image = list(os.listdir(folderName_image))
list_folderNameReceipt_csv = list(os.listdir(folderNameReceipt_csv))
list_folderName_json.sort()
adj_array = []
features_array = []


for json_file, image_file, receipt_csv_file in zip(list_folderName_json, list_folderName_image,
                                                   list_folderNameReceipt_csv):
    print(json_file, image_file, receipt_csv_file)
    receipt_csv = pd.read_csv(folderNameReceipt_csv + receipt_csv_file)
    img = cv2.imread(folderName_image + image_file, 0)

    # json_path = '/home/arjun/Gcn_paper/gcn/gcn/OneDrive_2020-11-13/spaCy NER Annotator output/Image_0006.json'
    tree = ObjectTree()
    tree.read(receipt_csv, img)

    graph_dict, text_list = tree.connect(plot=True, export_df=True)

    graph = Graph(max_nodes=len(text_list))

    adj, features = graph.make_graph_data(graph_dict, text_list)

    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    features = sparse.csr_matrix(features)
    with open(folderName_json + json_file) as f:
        data = json.load(f)

        q = []
        words = []
        q = []
        words = []
        for i in data['valid_line']:

            for j in    i['words']:

                #x_min = j['quad']['x1']
                #x_max = j['quad']['x2']
                #y_min = j['quad']['y2']
                #y_max = j['quad']['y4']
                #df_ = df_.append({'xmin': x_min,'ymin':y_min,'xmax':x_max,'ymax':y_max,'Object':j['text']},ignore_index = True)

                q.append(i['category'])

    a = np.zeros(shape=(len(set(q)), len(q))).T
    labels_df = pd.DataFrame(a, columns=set(q))

    for i in range(len(q)):
        labels_df[q[i]][i] = 1

    df_with_labels = pd.concat([receipt_csv, labels_df], axis=1)

    dataframe_collection['Image_' + str(Image_count)] = df_with_labels
    Image_count += 1


    adj, features, labels, idx_train, idx_val, idx_test = load_data(adj=adj, features=features, labels=df_with_labels)
    adj_array.append(adj)
    features_array.append(features)
    # df_with_labels = df_with_labels.sample(frac=1)

from scipy.sparse import block_diag

sparse_block_diag = torch.block_diag(adj_array[0],adj_array[1],adj_array[2])
print(sparse_block_diag)
raise

'''
###With one image
'''
receipt_csv = pd.read_csv('/home/arjun/receipts_by_cord/receipt_00000.csv')
#
img = cv2.imread('/home/arjun/Graph-Convolution-on-Structured-Documents/data/receipt_00000.png', 0)

# json_path = '/home/arjun/Gcn_paper/gcn/gcn/OneDrive_2020-11-13/spaCy NER Annotator output/Image_0006.json'
tree = ObjectTree()
tree.read(receipt_csv, img)

graph_dict, text_list = tree.connect(plot=True, export_df=True)

graph = Graph(max_nodes=len(text_list))

adj, features = graph.make_graph_data(graph_dict, text_list)
import json

with open('/home/arjun/receipts/json/receipt_00000.json') as f:
    data = json.load(f)

q = []
words = []
q = []
words = []
for i in data['valid_line']:

    for j in i['words']:

        #x_min = j['quad']['x1']
        #x_max = j['quad']['x2']
        #y_min = j['quad']['y2']
        #y_max = j['quad']['y4']
        #df_ = df_.append({'xmin': x_min,'ymin':y_min,'xmax':x_max,'ymax':y_max,'Object':j['text']},ignore_index = True)

        q.append(i['category'])


a = np.zeros(shape=(len(set(q)), len(q))).T
labels_df = pd.DataFrame(a, columns=set(q))

for i in range(len(q)):
    labels_df[q[i]][i] = 1

df_with_labels = pd.concat([receipt_csv, labels_df], axis=1)
df_with_labels = df_with_labels.sample(frac=1)
'''
## for 3 images
folderName_json = r"/home/arjun/receipts/json/"
folderName_image = r'/home/arjun/Graph-Convolution-on-Structured-Documents/data/image/'
folderNameReceipt_csv = '/home/arjun/receipts_by_cord/'

#folderName_json = r"/home/arjun/test_gcn/json/"
#folderName_image = r'/home/arjun/Graph-Convolution-on-Structured-Documents/data/test/'
#folderNameReceipt_csv = '/home/arjun/test_gcn/test/'
dataframe_collection = {}
Image_count = 0
list_folderName_json = list(sorted(os.listdir(folderName_json)))
list_folderName_image = list(sorted(os.listdir(folderName_image)))
list_folderNameReceipt_csv = list(sorted(os.listdir(folderNameReceipt_csv)))

adj_array = []
features_array = []
first_count = 0

count_image = 0
count_json = 0
count_csv = 0
name = 'receipt_'
image_missing = []
json_missing = []
csv_missing = []

##check and remove missing files
from pathlib import Path


def createList(r1, r2):
    return list(range(r1, r2 + 1))
'''

# Driver Code
r1, r2 = 0, 800
x = createList(r1, r2)

import random
random.shuffle(x)


for i in x:
    file_image = folderName_image + name + str(i).zfill(5) + '.png'
    file_image = Path(file_image)
    if file_image.is_file():
        flag = True
    else:
        count_image += 1
        image_missing.append(i)

    file_json = folderName_json + name + str(i).zfill(5) + '.json'
    file_json = Path(file_json)
    if file_json.is_file():
        flag = True
    else:
        count_json += 1
        json_missing.append(i)

    file_csv = folderNameReceipt_csv + name + str(i).zfill(5) + '.csv'
    file_csv = Path(file_csv)
    if file_csv.is_file():
        flag = True
    else:
        count_csv += 1
        csv_missing.append(i)

missing = 0
####

for i in x:
    if i in csv_missing or i in json_missing or i in image_missing:
        missing += 1
    else:

        json_file = name + str(i).zfill(5) + '.json'
        image_file = name + str(i).zfill(5) + '.png'
        receipt_csv_file = name + str(i).zfill(5) + '.csv'
        print(json_file, image_file, receipt_csv_file)

        receipt_csv = pd.read_csv(folderNameReceipt_csv + receipt_csv_file)
        img = cv2.imread(folderName_image + image_file, 0)

        # json_path = '/home/arjun/Gcn_paper/gcn/gcn/OneDrive_2020-11-13/spaCy NER Annotator output/Image_0006.json'
        tree = ObjectTree()
        tree.read(receipt_csv, img)

        graph_dict, text_list = tree.connect(plot=True, export_df=True)

        graph = Graph(max_nodes=len(text_list))
        adj, features = graph.make_graph_data(graph_dict, text_list)
        adj = sparse.csr_matrix(adj)
        adj = normalize(adj + sp.eye(adj.shape[0]))

        # adj = sparse_mx_to_torch_sparse_tensor(adj)
        if first_count == 0:
            features_merged = features
        else:
            features_merged = np.concatenate((features_merged, features), axis=0)

        # features = sparse.csr_matrix(features)
        # features = torch.FloatTensor(np.array(features.todense()))
        adj_array.append(adj.todense())

        features_array.append(features)
        with open(folderName_json + json_file) as f:
            data = json.load(f)

            q = []
            words = []
            q = []
            words = []
            for i in data['valid_line']:

                for j in i['words']:
                    q.append(i['category'])

        a = np.zeros(shape=(len(set(q)), len(q))).T
        labels_df = pd.DataFrame(a, columns=set(q))
        for i in range(len(q)):
            labels_df[q[i]][i] = 1

        df_with_labels = pd.concat([receipt_csv, labels_df], axis=1)
        if first_count == 0:
            df_merged_labels = df_with_labels
        else:
            frames = [df_merged_labels, df_with_labels]
            df_merged_labels = pd.concat(frames)
        dataframe_collection['Image_' + str(Image_count)] = df_with_labels
        Image_count += 1
        # df_with_labels = df_with_labels.sample(frac=1)
        first_count += 1
'''
#same_set_features_merged_20_pad_200_img_index_data_preprocess_int 53 lower and removal of punctuation
#same_set_features_merged_20_pad_200_img_index_data_robert_work_2_2  #best use 71
df_merged_labels = pd.read_csv('/home/arjun/same_set_df_merged_labels.csv')
df_merged_labels.iloc[:, 13:] = df_merged_labels.iloc[:, 13:].fillna(0)
features_merged = np.load('/home/arjun/same_set_features_merged_20_pad_200_img_index_data_robert_work_2_2.npy')
#features_merged = np.load('/media/arjun/Seagate Expansion Drive/gqp/features/same_set_features_merged_20_pad_200_img_index_robert.npy')
table_features = np.array(df_merged_labels[['xmin','ymin','xmax','ymax','revised_distances_vert','revised_distances_hori']])
features_merged = np.concatenate((features_merged,table_features),axis=1)
# data standardization with  sklearn

#features_merged_sparse = sparse.csr_matrix(features_merged)
features_merged_sparse = features_merged
#df_merged_labels.to_csv('df_merged_labels.csv', index=False)

## for the labels
cum_sum = 0
shape = []
'''for i in range(len(dataframe_collection)):
    shape.append(dataframe_collection['Image_' + str(i)].shape[0])

image_index = np.cumsum(shape)'''
image_index = np.load('/home/arjun/same_set_image_index.npy')
from scipy.sparse import block_diag
adj_array = list(np.load('/home/arjun/same_set_adj_array.npy',allow_pickle=True))
sparse_block_diag = block_diag(adj_array)

values = sparse_block_diag.data
indices = np.vstack((sparse_block_diag.row, sparse_block_diag.col))

i = torch.LongTensor(indices)
v = torch.FloatTensor(values)
shape = sparse_block_diag.shape

sp_adj = torch.sparse.FloatTensor(i, v, torch.Size(shape))
#
# features = sparse.csr_matrix(features)
# adj, features, labels, idx_train, idx_val, idx_test = load_data(adj=sp_adj,features=features_merged_sparse,labels=df_merged_labels,index=image_index)

adj, features, labels, idx_train, idx_val, idx_test = load_data(adj=sp_adj, features=features_merged_sparse,
                                                                labels=df_merged_labels, index=image_index)

# adj, features, labels, idx_train, idx_val, idx_test = load_data(adj=sp_adj,features=features_merged_sparse,labels=df_merged_labels)
# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.AdamW(model.parameters(),
                        lr=args.lr, weight_decay=args.weight_decay,eps=1e-4)

from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/same_set_gcn_cord')

# optimizer = optim.SparseAdam(model.parameters(),lr=args.lr)
if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
import random

def train(epoch):
    t = time.time()

    model.train()
    optimizer.zero_grad()
    global adj, features, labels, idx_train, idx_val, idx_test
    if epoch%1000==0 and epoch>0:
        global adj, features, labels, idx_train, idx_val, idx_test

        global features_merged
        idx = np.random.permutation(pd.DataFrame(features_merged)[0:16174].index)
        features_merged_stored_separately = features_merged[16174:]
        features_merged = np.concatenate((features_merged[0:16174][idx,:],features_merged_stored_separately))
        global df_merged_labels
        df_merged_labels_stored_sep = df_merged_labels[16174:]
        df_merged_labels = pd.concat([df_merged_labels[0:16174].iloc[idx, :], df_merged_labels_stored_sep])

        global sp_adj
        sp_adj_stored_sep = sp_adj.to_dense()[16174:]
        sam = sparse.coo_matrix(torch.cat([sp_adj.to_dense()[0:16174][idx, :], sp_adj_stored_sep], 0))
        values = sam.data
        indices = np.vstack((sam.row, sam.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = sam.shape

        sp_adj = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        adj, features, labels, idx_train, idx_val, idx_test = load_data(adj=sp_adj, features=features_merged,
                                                                        labels=df_merged_labels, index=image_index)
        if args.cuda:
            model.cuda()
            features = features.cuda()
            adj = adj.cuda()
            labels = labels.cuda()
            idx_train = idx_train.cuda()
            idx_val = idx_val.cuda()
            idx_test = idx_test.cuda()

    output = model(features, adj)

    #print(output)
    #print(idx_train)
    #print(output[idx_train])
    loss_train = F.cross_entropy(output[idx_train], labels[idx_train])

    writer.add_scalar('Loss/train', loss_train, epoch)

    print('acc_train')
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)


    loss_val = F.cross_entropy(output[idx_val], labels[idx_val])

    print('acc_val')
    writer.add_scalar('Loss/val', loss_val, epoch)
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    '''
    adj, features, labels, idx_train, idx_val, idx_test = load_data(adj=sp_adj, features=features_merged_sparse,
                                                                    labels=df_merged_labels, index=image_index,check='Test')
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout)
    model.load_state_dict(torch.load('/home/arjun/same_set_best_72_robert_40_text_int_22.pt'))

    #model.load_state_dict(torch.load('/media/arjun/Seagate Expansion Drive/gqp/features/'))
    '''
    model.eval()

    output = model(features, adj)

    loss_test = F.cross_entropy(output[idx_test], labels[idx_test])

    # print(output[idx_test],'output[idx_test]')
    # print(labels_df.columns)
    print(labels[idx_test], 'labels[idx_test]')
    # writer.add_scalar('Loss/test', loss_test, epoch)

    acc_test = accuracy(output[idx_test], labels[idx_test],flag=False,target_names= df_merged_labels.iloc[:, 13:].columns)
    acc_train = accuracy(output[idx_train], labels[idx_train], flag=False,
                        target_names=df_merged_labels.iloc[:, 13:].columns)


    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
#print(df_merged_labels.iloc[idx_test.detach().cpu(), 4], 'test')
#torch.save(model.state_dict(), '/home/arjun/same_set_best_72_robert_40_text_int_22.pt') best use 71
#torch.save(model.state_dict(), '/home/arjun/same_set_best_72_robert_40_text_int_22_symmetric_adj_array.pt') #68
torch.save(model.state_dict(), '/home/arjun/same_set_best_72_robert_40_text_int_22_symmetric_adj_array_stan_rand.pt')
#torch.save(model.state_dict(), '/home/arjun/same_set_best_72_robert_40_text_int_22_preprocess_no_preprocess_int.pt') worst  porter 69
# Testing
test()

