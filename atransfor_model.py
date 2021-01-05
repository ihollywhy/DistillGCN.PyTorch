import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn

msg_func = fn.copy_src(src='h', out='m')
reduce_func = fn.sum(msg='m', out='h')

class GCNLayer(nn.Module):
    """
    a GCN model to transfer the feature of student to the feature of teacher
    """
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.nonlinear = nn.LeakyReLU(negative_slope=0.2)
        self.batchnorm = nn.BatchNorm1d(out_feats)

    def apply(self, nodes):
        return {'h': self.linear(nodes.data['h'])}

    def forward(self, g, feature):
        feature = self.nonlinear(feature)
        g.ndata['h'] = feature
        g.update_all(msg_func, reduce_func)
        g.apply_nodes(func=self.apply)
        return self.batchnorm(g.ndata.pop('h'))


def get_gcn_transformer(args, feat_info):
    input_dim = feat_info['s_feat'][0]
    output_dim = feat_info['t_feat'][0]
    gcn_transformer = GCNLayer(input_dim, output_dim)
    return gcn_transformer


def get_transformer_model(args, feat_info):
    """transfer the student feature to teacher feature
    use for comparing the features directly
    """
    class transformer_model(nn.Module):
        def __init__(self, in_dim, out_dim):
            super(transformer_model, self).__init__()
            self.relu = nn.ReLU()
            self.transfer = nn.Linear(in_dim, out_dim)
        
        def forward(self,x):
            # x = self.relu(x)
            return self.transfer(x)

    input_dim = feat_info['s_feat'][0]
    output_dim = feat_info['t_feat'][0]
    return transformer_model(input_dim, output_dim)


if __name__ == '__main__':
    print(get_gcn_transformer(4,6))
    
