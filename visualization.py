# provide a t-sne vis for the embedded features

import os
import copy
import numpy as np
import torch
import torch.nn.functional as F

import dgl
import argparse
from gat import GAT
from utils import evaluate, collate
from utils import get_data_loader, save_checkpoint, load_checkpoint
from utils import evaluate_model, test_model, generate_label
from auxilary_loss import gen_fit_loss, optimizing, gen_mi_loss, loss_fn_kd, gen_att_loss
from auxilary_model import collect_model
from auxilary_optimizer import block_optimizer
from plot_utils import loss_logger, parameters
import time
import matplotlib.pyplot as plt
import collections
import random

import pickle
from sklearn import manifold


torch.set_num_threads(1)

def vis_feat():
    
    with open('middle_res/t_model_feat.pkl', 'rb') as f:
        (X, labels) = pickle.load(f)

    N = X.shape[0]
    index0 = np.array(list(range(N)))
    np.random.shuffle(index0)

    index1 = np.array(list(range(N)))
    np.random.shuffle(index1)

    distance_feat = np.sum(np.abs(X[index0] - X[index1]), axis=-1)
    distance_label = np.sum(np.abs(labels[index0] - labels[index1]), axis=-1)
    print(distance_feat.shape)
    plt.scatter(distance_feat, distance_label)
    plt.show()


def vis_feat_old():
    with open('middle_res/t_model_feat.pkl', 'rb') as f:
        (X, labels) = pickle.load(f)


    tsne = manifold.TSNE(n_components=2, init='random',
                         random_state=0, perplexity=50)
    
    N = X.shape[0]
    index = np.array(list(range(N)))

    print(X.shape)
    print(labels.shape)
    idx = 100
    sampleN = 4000
    np.random.shuffle(index)
    index = index[:sampleN]
    X = X[index,:]
    Y = labels[index, idx]
    #print(Y)
    colors = ['m']*sampleN
    
    from matplotlib import cm
    cmap_name = 'Blues'
    cmap = cm.get_cmap(cmap_name)
    
    distances = []
    for i in range(sampleN):
        distance = np.sum(np.abs(labels[i,:] - labels[0,:])) / 121.0
        distances.append(distance)
    distances -= np.min(distances)
    distances /= np.max(distances)
    
    colors = []
    for distance in distances:
        colors.append(cmap(distance))

    X = tsne.fit_transform( X )
    f = plt.figure(figsize=(5,5))
    ax = f.add_subplot(111)

    ax.scatter(X[:, 0], X[:, 1], c=colors)
    #ax.scatter(X[index0, 0], X[index0, 1], c='m')
    #ax.scatter(X[index1, 0], X[index1, 1], c='c')
    ax.axis('tight')
    plt.show()


def vis(args, auxiliary_model, data, device):
    train_dataloader, valid_dataloader, test_dataloader, fixed_train_dataloader = data
    
    t_model = auxiliary_model['t_model']['model']
    s_model_full = auxiliary_model['s_model']['model']
    s_model_lsp = copy.deepcopy( auxiliary_model['s_model']['model'] )
    
    load_checkpoint(s_model_full, f'./models/s_model_full_{args.load_epoch}.pt' ,device)
    load_checkpoint(s_model_lsp, f'./models/s_model_lsp_{args.load_epoch}.pt' ,device)
    
    
    s_model_full.eval()
    s_model_lsp.eval()
    t_model.eval()
    for batch, batch_data in enumerate( zip(train_dataloader,fixed_train_dataloader) ):
        shuffle_data, fixed_data = batch_data
        subgraph, feats, labels = shuffle_data
        fixed_subgraph, fixed_feats, fixed_labels = fixed_data

        feats = feats.to(device)
        labels = labels.to(device)
        fixed_feats = fixed_feats.to(device)
        fixed_labels = fixed_labels.to(device)

        t_model.g = subgraph
        for layer in t_model.gat_layers:
            layer.g = subgraph
        
        _, middle_feats_t = t_model(feats.float(), middle=True)

        print(middle_feats_t[-1].shape)
    
    labels = labels.detach().cpu().numpy()
    X = middle_feats_t[-1].detach().cpu().numpy()
    
    with open('middle_res/t_model_feat.pkl', 'wb') as f:
        pickle.dump((X, labels), f)

    
def main(args):
    device = torch.device("cpu") if args.gpu<0 else torch.device("cuda:" + str(args.gpu))
    data, data_info = get_data_loader(args)
    model_dict = collect_model(args, data_info)

    t_model = model_dict['t_model']['model']
    # load or train the teacher
    load_checkpoint(t_model, "./models/t_model.pt", device)

    print("############ train student with teacher #############")
    vis(args, model_dict, data, device)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument("--gpu", type=int, default=1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--residual", action="store_true", default=True,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=0,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=0,
                        help="attention dropout")
    parser.add_argument('--alpha', type=float, default=0.2,
                        help="the negative slop of leaky relu")
    parser.add_argument('--batch-size', type=int, default=2,
                        help="batch size used for training, validation and test")


    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=0,
                        help="weight decay")

    parser.add_argument("--t-epochs", type=int, default=60,
                        help="number of training epochs")
    parser.add_argument("--t-num-heads", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--t-num-out-heads", type=int, default=6,
                        help="number of output attention heads")
    parser.add_argument("--t-num-layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--t-num-hidden", type=int, default=256,
                        help="number of hidden units")

    parser.add_argument("--s-epochs", type=int, default=500,
                        help="number of training epochs")
    parser.add_argument("--s-num-heads", type=int, default=2,
                        help="number of hidden attention heads")
    parser.add_argument("--s-num-out-heads", type=int, default=2,
                        help="number of output attention heads")
    parser.add_argument("--s-num-layers", type=int, default=4,
                        help="number of hidden layers")
    parser.add_argument("--s-num-hidden", type=int, default=68,
                        help="number of hidden units")
    parser.add_argument("--target-layer", type=int, default=2,
                        help="the layer of student to learn")
    
    parser.add_argument("--mode", type=str, default='mi',
                        help="model used: teacher, full, mi, fitnets")
    parser.add_argument("--train-mode", type=str, default='together',
                        help="training mode: together, warmup")
    parser.add_argument("--warmup-epoch", type=int, default=600,
                        help="steps to warmup")

    parser.add_argument('--loss-weight', type=float, default=1.0,
                        help="weight coeff of additional loss")
    
    parser.add_argument('--load_epoch', type=int, default=5,
                        help="epoch for student mode loading")

    args = parser.parse_args()
    print(args)

    torch.manual_seed(100)
    torch.cuda.manual_seed(100)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    vis_feat()
    #main(args)
