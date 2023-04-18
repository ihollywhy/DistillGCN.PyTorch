import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn

from sklearn.metrics import f1_score
import dgl
from dgl.data.ppi import LegacyPPIDataset as PPIDataset
from gat import GAT, GCN
from auxilary_loss import gen_mi_loss


class FullLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.regular_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, logits, labels, inputs, graph, middle_feats_s, target, loss_weight, t_model):
        loss = self.regular_loss(logits, labels)
        return loss, torch.tensor([0])


class TeacherLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.regular_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, logits, labels, inputs, graph, middle_feats_s, target, loss_weight, t_model):
        loss = self.regular_loss(logits, labels)
        return loss, torch.tensor([0])

class MILoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.regular_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, logits, labels, graph, inputs, middle_feats_s, target, loss_weight, t_model):
        reg_loss = self.regular_loss(logits, labels)
        #logits_t = t_model(graph, inputs)
        lsp_loss = gen_mi_loss(t_model, middle_feats_s[target], graph, inputs)
        return reg_loss + loss_weight * lsp_loss, lsp_loss


def train_epoch(train_dataloader, model, loss_fcn, optimizer, device):
    model.train()
    loss_list = []
    for batch, batch_data in enumerate(train_dataloader):
        # getting the data
        subgraph, feats, labels = batch_data
        feats, labels = feats.to(device), labels.to(device)

        # forward pass
        logits = model(subgraph, feats.float())
        loss = loss_fcn(logits, labels.float())
        
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())
    return np.array(loss_list).mean()


def evaluate(dataloader, model, loss_func, device):
    model.eval()
    score_list = []
    val_loss_list = []
    for batch, graph_data in enumerate(dataloader):
        # getting data
        subgraph, feats, labels = graph_data
        feats, labels = feats.to(device), labels.to(device)
        
        # forward
        with torch.no_grad():
            output = model(subgraph, feats.float())
            val_loss = loss_func(output, labels.float())
            predict = np.where(output.data.cpu().numpy() >= 0.5, 1, 0)
            score = f1_score(labels.data.cpu().numpy(), predict, average='micro')

        # storing stats
        score_list.append(score)
        val_loss_list.append(val_loss.item())
    mean_score = np.array(score_list).mean()
    mean_val_loss = np.array(val_loss_list).mean()
    return mean_score, mean_val_loss


def train_epoch_s(train_dataloader, model, model_t,loss_fcn, optimizer, device, args):
    model.train()
    loss_list = []
    lsp_loss_list = []
    for batch, batch_data in enumerate(train_dataloader):
        # getting the data
        subgraph, feats, labels = batch_data
        feats, labels = feats.to(device), labels.to(device)

        # forward pass
        logits, middle_feats_s = model(subgraph, feats.float(), middle=True)
        loss, lsp_loss = loss_fcn(logits, labels.float(), subgraph, feats.float(), middle_feats_s, args.target_layer, args.loss_weight, model_t)
        
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())
        lsp_loss_list.append(lsp_loss.item())
    return np.array(loss_list).mean(), np.array(lsp_loss_list).mean()


def generate_label(t_model, subgraph, feats, device):
    '''generate pseudo lables given a teacher model
    '''
    # t_model.to(device)
    t_model.eval()
    with torch.no_grad():
        # soft labels
        logits_t = t_model(subgraph, feats.float())
        #pseudo_labels = torch.where(t_logits>0.5, 
        #                            torch.ones(t_logits.shape).to(device), 
        #                            torch.zeros(t_logits.shape).to(device))
        #labels = logits_t
    return logits_t.detach()
    

def collate(sample):
    graphs, feats, labels =map(list, zip(*sample))
    graph = dgl.batch(graphs)
    feats = torch.from_numpy(np.concatenate(feats))
    labels = torch.from_numpy(np.concatenate(labels))
    return graph, feats, labels

def collate_w_gk(sample):
    '''
    collate with graph_khop
    '''
    graphs, feats, labels, graphs_gk =map(list, zip(*sample))
    graph = dgl.batch(graphs)
    graph_gk = dgl.batch(graphs_gk)
    feats = torch.from_numpy(np.concatenate(feats))
    labels = torch.from_numpy(np.concatenate(labels))
    return graph, feats, labels, graph_gk


def get_teacher(args, data_info):
    '''args holds the common arguments
    data_info holds some special arugments
    '''
    heads = ([args.t_num_heads] * args.t_num_layers) + [args.t_num_out_heads]
    model = GAT(args.t_num_layers,
            data_info['num_feats'],
            args.t_num_hidden,
            data_info['n_classes'],
            heads,
            F.elu,
            args.in_drop,
            args.attn_drop,
            args.alpha,
            args.residual)
    return model
    
def get_student(args, data_info):
    '''args holds the common arguments
    data_info holds some special arugments
    '''
    heads = ([args.s_num_heads] * args.s_num_layers) + [args.s_num_out_heads]
    model = GAT(args.s_num_layers,
            data_info['num_feats'],
            args.s_num_hidden,
            data_info['n_classes'],
            heads,
            F.elu,
            args.in_drop,
            args.attn_drop,
            args.alpha,
            args.residual)
    return model

def get_feat_info(args):
    feat_info = {}
    # list multpilication [3] * 3 == [3,3,3]
    feat_info['s_feat'] = [args.s_num_heads*args.s_num_hidden] * args.s_num_layers
    feat_info['t_feat'] = [args.t_num_heads*args.t_num_hidden] * args.t_num_layers
    #assert len(feat_info['s_feat']) == len(feat_info['t_feat']),"number of hidden layer for teacher and student are not equal"
    return feat_info


def get_data_loader(args):
    '''create the dataset
    return 
        three dataloders and data_info
    '''
    train_dataset = PPIDataset(mode='train')
    valid_dataset = PPIDataset(mode='valid')
    test_dataset = PPIDataset(mode='test')
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate, num_workers=4, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=collate, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate, num_workers=2)

    n_classes = train_dataset.labels.shape[1]
    num_feats = train_dataset.features.shape[1]
    g = train_dataset.graph
    data_info = {}
    data_info['n_classes'] = n_classes
    data_info['num_feats'] = num_feats
    data_info['g'] = g
    return (train_dataloader, valid_dataloader, test_dataloader), data_info


def save_checkpoint(model, path):
    '''Saves model
    '''
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    torch.save(model.state_dict(), path)
    print(f"save model to {path}")

def load_checkpoint(model, path, device):
    '''load model
    '''
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"Load model from {path}")

