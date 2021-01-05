import torch


def block_optimizer(args, auxiliary_model, model_name, blocks_lr):
    model = auxiliary_model[model_name]["model"]
    group = [
        {"params": model.gat_layers[0].parameters(), 'lr': blocks_lr[0]},
        {"params": model.gat_layers[1].parameters(), 'lr': blocks_lr[1]},
        {"params": model.gat_layers[2].parameters(), 'lr': blocks_lr[2]},
        {"params": model.gat_layers[3].parameters(), 'lr': blocks_lr[3]},
        {"params": model.gat_layers[4].parameters(), 'lr': blocks_lr[4]}
    ] 
    auxiliary_model[model_name]['optimizer'] = torch.optim.Adam(group, lr=args.lr, weight_decay=args.weight_decay)
