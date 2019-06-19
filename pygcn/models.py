import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution,BI_Intereaction
import torch

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.bi2 = BI_Intereaction(nfeat, 8)
        self.fc1 = nn.Linear(8, nhid)
        self.dropout = dropout
        self.bn = nn.BatchNorm1d(8)

    def forward(self, x, adj):

        # x = F.relu(self.gc1(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc2(x, adj)

        x_left = F.relu(self.gc1(x, adj))
        x_left = F.dropout(x_left, self.dropout, training=self.training)


        x_right = F.relu(self.bi2(x))
        x_right = self.bn(x_right)
        # x_right = self.fc1(x_right)

        x = torch.cat((x_left, x_right), 1 )
        x = ( x_left + x_right ) * 0.5

        x = self.gc2(x,adj)
        return F.log_softmax(x, dim=1)
