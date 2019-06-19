import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution,BI_Intereaction
import torch

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, n_embedding, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(2 * nhid, nclass)


        self.bi1 = BI_Intereaction(nfeat, 2 * n_embedding)
        self.fc1 = nn.Linear(2 * n_embedding, n_embedding)

        self.fc2 = nn.Linear(n_embedding,8)
        self.relu1 = nn.ReLU()

        self.net = nn.Sequential(self.bi1, self.fc1, self.relu1, self.fc2)

        self.dropout = dropout
        self.bn = nn.BatchNorm1d(8)

    def forward(self, x, adj):

        # x = F.relu(self.gc1(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc2(x, adj)

        x_left = F.relu(self.gc1(x, adj))
        x_left = F.dropout(x_left, self.dropout, training=self.training)


        x_right = F.relu(self.net(x))
        x_right = self.bn(x_right)
        # x_right = self.fc1(x_right)

        x = torch.cat((x_left, x_right), 1 )
        # x = ( x_left + x_right ) * 0.5

        x = self.gc2(x,adj)
        return F.log_softmax(x, dim=1)
