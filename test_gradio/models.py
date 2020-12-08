import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)

        self.bn = nn.BatchNorm1d(256)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        #print(x.shape,'x')
        #print(adj.shape,'adj')
        x = F.leaky_relu(self.gc1(x, adj))
        x = self.bn(x)
        embeddings = x
        x = F.dropout(x, self.dropout, training=self.training)
        #x = self.bn(x)

        x = self.gc2(x, adj)
        #return F.log_softmax(x, dim=30)
        return x
