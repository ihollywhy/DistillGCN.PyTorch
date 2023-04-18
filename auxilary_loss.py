import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import statistics_feature
from dgl.nn.pytorch.softmax import edge_softmax
import dgl
import dgl.function as fn
from local_structure import get_graph_local_structure


statistics_plot = statistics_feature()

class loss_scheduler():
    def __init__(self, args, warmup=100):
        self.loss_weight = args.loss_weight

    def get_loss(self, *losses, epoch):
        return 0

def graph_KLDiv(graph, edgex, edgey, reduce='mean'):
    '''
    compute the KL loss for each edges set, used after edge_softmax
    '''
    with graph.local_scope():
        nnode = graph.number_of_nodes()
        graph.ndata.update({'kldiv': torch.ones(nnode,1).to(edgex.device)})
        diff = edgey*(torch.log(edgey)-torch.log(edgex))
        graph.edata.update({'diff':diff})
        graph.update_all(fn.u_mul_e('kldiv', 'diff', 'm'),
                            fn.sum('m', 'kldiv'))
        if reduce == "mean":
            return torch.mean(torch.flatten(graph.ndata['kldiv']))
    

def loss_fn_kd(logits, logits_t, alpha=1.0, T=10.0):
    """
    logits: pre-softmax or sigmoid activation output of student
    logits_t: pre-softmax or sigmoid activation output of teacher
    """
    ce_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    kl_loss_fn = nn.KLDivLoss()
    mse_loss_fn = nn.MSELoss()
    labels_t = torch.where(logits_t>0.0, 
                        torch.ones(logits_t.shape).to(logits_t.device), 
                        torch.zeros(logits_t.shape).to(logits_t.device))
    ce_loss = ce_loss_fn(logits, labels_t)
    return ce_loss
    d_s = torch.log( torch.cat((torch.sigmoid(logits/T), 1-torch.sigmoid(logits/T)), dim=1) )
    d_t = torch.cat((torch.sigmoid(logits_t/T), 1-torch.sigmoid(logits_t/T)), dim=1)
    kl_loss = kl_loss_fn(d_s , d_t)*T*T
    #mse_loss = mse_loss_fn(logits, logits_t)
    return ce_loss*alpha + (1-alpha)*kl_loss

def gen_mi_loss(t_model, middle_feats_s, subgraph, feats):
    """
    Params:
        middle_feats_s  -   student's middle features
        subgraph  -  subgraph of feats
        feats  -  the input features
        device - pytorch device
    """
    with torch.no_grad():
        _, middle_feats_t = t_model(subgraph, feats.float(), middle=True)
        middle_feats_t = middle_feats_t[1]
    
    dist_t = get_graph_local_structure(subgraph, middle_feats_t)
    dist_s = get_graph_local_structure(subgraph, middle_feats_s)
    graphKL_loss = graph_KLDiv(subgraph, dist_s, dist_t)
    return graphKL_loss


def gen_att_loss(auxiliary_model, middle_feats_s, subgraph, feats, device):
    """
    generate the loss according to a similar stratagy shown in attention transfer paper
    """
    loss_fcn = nn.MSELoss()
    
    t_model = auxiliary_model['t_model']['model']
    
    with torch.no_grad():
        _, middle_feats_t = t_model(subgraph, feats.float(), middle=True)
        middle_feats_t = middle_feats_t[1].detach()
    
    middle_feats_t = torch.abs(middle_feats_t)
    middle_feats_t = torch.mean(middle_feats_t, dim=-1)
    
    middle_feats_s = torch.abs(middle_feats_s)
    middle_feats_s = torch.mean(middle_feats_s, dim=-1)
    
    return loss_fcn(middle_feats_s, middle_feats_t)

## problem
def gen_fit_loss(auxiliary_model, middle_feats_s, subgraph, feats, device):
    """
    generate the loss according to a similar stratagy shown in fitnets paper
    """
    loss_fcn = nn.MSELoss()
    
    upsampled_feats_s = auxiliary_model['upsampling_model']['model'](subgraph, middle_feats_s)

    t_model = auxiliary_model['t_model']['model']
    
    with torch.no_grad():
        _, middle_feats_t = t_model(subgraph, feats.float(), middle=True)
        middle_feats_t = middle_feats_t[1].detach()
    
    return loss_fcn(upsampled_feats_s, middle_feats_t)

def optimizing(auxiliary_model, loss, model_list):
    """
    args: 
        auxiliary_model: dict of dict [model_name][model/optimizer]
        loss:
        model_list: the name of models need to be updated
    """
    for model in model_list:
        auxiliary_model[model]['optimizer'].zero_grad()
    
    loss.backward()
    
    for model in model_list:
        auxiliary_model[model]['optimizer'].step()

if __name__ == "__main__":
    g = dgl.DGLGraph()
    g.add_nodes(3)
    g.ndata.update({'kldiv':torch.ones(3,1)})
    g.add_edges([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2])
    edata = torch.ones(6, 1).float()
    p = edge_softmax(g, edata)
    edata = torch.Tensor([[1],[2],[3],[4],[5],[6]]).float()
    q = edge_softmax(g, edata)
    diff = p*(torch.log(q)-torch.log(p))
    g.edata.update({'diff':diff})
    print(g.edata)
    g.update_all(fn.u_mul_e('kldiv', 'diff', 'm'),
                         fn.sum('m', 'kldiv'))
    print(g.ndata)
