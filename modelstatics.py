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

torch.set_num_threads(1)

def train_student(args, auxiliary_model, data, device):
    best_score = 0

    train_dataloader, valid_dataloader, test_dataloader, fixed_train_dataloader = data
    
    # multi class loss function
    loss_fcn = torch.nn.BCEWithLogitsLoss()
    loss_mse = torch.nn.MSELoss()

    t_model = auxiliary_model['t_model']['model']
    s_model = auxiliary_model['s_model']['model']
    
    losslogger = loss_logger()
    step_n = 0
    has_run = False
    start = time.time()
    for i in range(100):
        with torch.no_grad():
            for batch, batch_data in enumerate( zip(train_dataloader,fixed_train_dataloader) ):
                step_n += 1
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
                logits = t_model(feats.float())
                """
                s_model.g = subgraph
                for layer in s_model.gat_layers:
                    layer.g = subgraph
                logits = s_model(feats.float())
                """
    print(f"time: {time.time()-start}")

def train_teacher(args, model, data, device):
    train_dataloader, valid_dataloader, test_dataloader, _ = data
    
    best_model = None
    best_val = 0
    # define loss function
    loss_fcn = torch.nn.BCEWithLogitsLoss()
    
    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    for epoch in range(args.t_epochs):
        model.train()
        loss_list = []
        for batch, batch_data in enumerate(train_dataloader):
            subgraph, feats, labels = batch_data
            feats = feats.to(device)
            labels = labels.to(device)
            model.g = subgraph
            for layer in model.gat_layers:
                layer.g = subgraph
            logits = model(feats.float())
            loss = loss_fcn(logits, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        loss_data = np.array(loss_list).mean()
        print(f"Epoch {epoch + 1:05d} | Loss: {loss_data:.4f}")
        if epoch % 10 == 0:
            score_list = []
            val_loss_list = []
            for batch, valid_data in enumerate(valid_dataloader):
                subgraph, feats, labels = valid_data
                feats = feats.to(device)
                labels = labels.to(device)
                score, val_loss = evaluate(feats.float(), model, subgraph, labels.float(), loss_fcn)
                score_list.append(score)
                val_loss_list.append(val_loss)
            mean_score = np.array(score_list).mean()
            mean_val_loss = np.array(val_loss_list).mean()
            print(f"F1-Score on valset  :        {mean_score:.4f} ")
            if mean_score > best_val:
                best_model = copy.deepcopy(model)

            train_score_list = []
            for batch, train_data in enumerate(train_dataloader):
                subgraph, feats, labels = train_data
                feats = feats.to(device)
                labels = labels.to(device)
                train_score_list.append(evaluate(feats, model, subgraph, labels.float(), loss_fcn)[0])
            print(f"F1-Score on trainset:        {np.array(train_score_list).mean():.4f}")

    # model = best_model

    test_score_list = []
    for batch, test_data in enumerate(test_dataloader):
        subgraph, feats, labels = test_data
        feats = feats.to(device)
        labels = labels.to(device)
        test_score_list.append(evaluate(feats, model, subgraph, labels.float(), loss_fcn)[0])
    print(f"F1-Score on testset:        {np.array(test_score_list).mean():.4f}")
    


def main(args):
    device = torch.device("cpu") if args.gpu<0 else torch.device("cuda:" + str(args.gpu))
    data, data_info = get_data_loader(args)
    model_dict = collect_model(args, data_info)

    t_model = model_dict['t_model']['model']

    # load or train the teacher
    if os.path.isfile("./models/t_model.pt"):
        load_checkpoint(t_model, "./models/t_model.pt", device)
    else:
        print("############ train teacher #############")
        train_teacher(args, t_model, data, device)
        save_checkpoint(t_model, "./models/t_model.pt")
    

    print(f"number of parameter for teacher model: {parameters(t_model)}")
    print(f"number of parameter for student model: {parameters(model_dict['s_model']['model'])}")

    # verify the teacher model
    loss_fcn = torch.nn.BCEWithLogitsLoss()
    train_dataloader, _, test_dataloader, _ = data
    print(f"test acc of teacher:")
    test_model(test_dataloader, t_model, device, loss_fcn)
    print(f"train acc of teacher:")
    test_model(train_dataloader, t_model, device, loss_fcn)
    

    print("############ train student with teacher #############")
    train_student(args, model_dict, data, device)



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

    args = parser.parse_args()
    print(args)

    torch.manual_seed(100)
    torch.cuda.manual_seed(100)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    main(args)
