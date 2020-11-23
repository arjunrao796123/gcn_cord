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
from grapher import ObjectTree, Graph
import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from utils import load_data, accuracy
from models import GCN
import os
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


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
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=20000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.3,
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

for i in range(800):
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

missing=0
####
for i in range(800):
    if i in csv_missing or i in json_missing or i in image_missing:
        missing+=1
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
                    '''
                    x_min = j['quad']['x1']
                    x_max = j['quad']['x2']
                    y_min = j['quad']['y2']
                    y_max = j['quad']['y4']
                    df_ = df_.append({'xmin': x_min,'ymin':y_min,'xmax':x_max,'ymax':y_max,'Object':j['text']},ignore_index = True)
                    '''
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

df_merged_labels.iloc[:, 13:] = df_merged_labels.iloc[:, 13:].fillna(0)
features_merged_sparse = sparse.csr_matrix(features_merged)

from scipy.sparse import block_diag
sparse_block_diag = block_diag(adj_array)


values = sparse_block_diag.data
indices = np.vstack((sparse_block_diag.row, sparse_block_diag.col))

i = torch.LongTensor(indices)
v = torch.FloatTensor(values)
shape = sparse_block_diag.shape

sp_adj = torch.sparse.FloatTensor(i, v, torch.Size(shape))
#
#features = sparse.csr_matrix(features)
adj, features, labels, idx_train, idx_val, idx_test = load_data(adj=sp_adj,features=features_merged_sparse,labels=df_merged_labels)


# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    print(features.shape)
    print(adj.shape)
    output = model(features, adj)
    print(output)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    print('acc_train')
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    print('acc_val')
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)

    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    print(output[idx_test],'output[idx_test]')
    print(labels_df.columns)
    print(labels[idx_test],'labels[idx_test]')
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
