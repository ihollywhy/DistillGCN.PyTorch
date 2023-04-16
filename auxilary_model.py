import torch
from utils import get_teacher, get_student

def collect_model(args, data_info):
    device = torch.device("cpu") if args.gpu<0 else torch.device("cuda:" + str(args.gpu))
    #feat_info = get_feat_info(args)
    
    # construct models
    t_model = get_teacher(args, data_info).to(device)                         
    s_model = get_student(args, data_info).to(device)                         
    #upsampling_model = get_upsampling_model(feat_info).to(device);


    # construct optimizers
    s_model_optimizer = torch.optim.Adam(s_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    t_model_optimizer = torch.optim.Adam(t_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #upsampling_model_optimizer = torch.optim.Adam(upsampling_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # construct model dict
    model_dict = {}
    model_dict['s_model'] = {'model':s_model, 'optimizer':s_model_optimizer}
    model_dict['t_model'] = {'model':t_model, 'optimizer':t_model_optimizer}
    #model_dict['upsampling_model'] = {'model':upsampling_model, 'optimizer': upsampling_model_optimizer}
    return model_dict