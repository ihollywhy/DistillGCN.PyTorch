import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch.softmax import edge_softmax


def _compute_local_feats(middle_feats, g, mode='mean'):
    '''get local feature, 
    one can also compute it through a gcn layer
    given:
        middle_feats and graph
    '''
    graph = g.local_var()
    graph.ndata['h'] = middle_feats
    if mode == 'max':
        graph.update_all(fn.copy_src('h', 'm'), fn.max('m', 'neigh'))
    elif mode == 'mean':
        graph.update_all(fn.copy_src('h', 'm'), fn.mean('m', 'neigh'))
    local_feats = graph.ndata['neigh']
    return local_feats

class customized_GAT(nn.Module):
    def __init__(self, in_dim, out_dim, retatt=False):
        super(customized_GAT, self).__init__()
        self.GATConv1 = GATConv(in_feats=in_dim, out_feats=out_dim, num_heads=1)
        self.nonlinear = nn.LeakyReLU(negative_slope=0.2)
        self.retatt = retatt

    def forward(self, graph, feats):
        feats = self.nonlinear(feats)
        if self.retatt:
            rst, att = self.GATConv1(graph, feats, self.retatt)
            return rst.flatten(1), att
        else:
            rst = self.GATConv1(graph, feats, self.retatt)
            return rst.flatten(1)

class distanceNet(nn.Module):
    def __init__(self):
        super(distanceNet, self).__init__()
        
    def forward(self, graph, feats):
        graph = graph.local_var()
        feats = feats.view(-1, 1, feats.shape[1])
        graph.ndata.update({'ftl': feats, 'ftr': feats})
        # compute edge distance

        # gaussion
        graph.apply_edges(fn.u_sub_v('ftl', 'ftr', 'diff'))
        e = graph.edata.pop('diff')
        e = torch.exp( (-1.0/100) * torch.sum(torch.abs(e), dim=-1) )
        
        # compute softmax
        e = edge_softmax(graph, e)
        return e


def old_get_local_model(feat_info, upsampling=False):
    '''model to compute a local feature given a graph and features
    retatt: return attention coefficients and donot apply linear transformation
    '''
    if upsampling:
        return customized_GAT(feat_info['s_feat'][1], feat_info['t_feat'][1], retatt=True)
    return customized_GAT(feat_info['t_feat'][1], feat_info['t_feat'][1], retatt=True)


def get_graph_local_structure(graph, feats):
        with graph.local_scope():
            feats = feats.view(-1, 1, feats.shape[1])
            graph.ndata.update({'ftl': feats, 'ftr': feats})
            graph.apply_edges(fn.u_sub_v('ftl', 'ftr', 'diff'))
            e = graph.edata.pop('diff')
            e = torch.exp( (-1.0/100) * torch.sum(torch.abs(e), dim=-1) )
            e = edge_softmax(graph, e)
        return e


def get_upsampling_model(feat_info):
    '''upsampling the features of a graph
    '''
    return customized_GAT(feat_info['s_feat'][1],feat_info['t_feat'][1])