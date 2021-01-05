import torch
import torch.nn as nn
import torch.nn.functional as F


class discriminator_model(nn.Module):
    """independent discriminator model for student
    it will first trained using the teacher's feature
    and then added as an additional loss to the student
    args:
        input_dim - the input dimension
    """
    def __init__(self, input_dim):
        super(discriminator_model, self).__init__()
        self.layers = nn.Sequential(
                    nn.Linear(input_dim, input_dim//2),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Linear(input_dim//2, input_dim//4),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Linear(input_dim//4, 1)
                    )
    
    def forward(self, x):
        return F.sigmoid(self.layers(x))


def get_discriminator(args, feat_info):
    input_dim = feat_info['t_feat'][1]*2
    discriminator = discriminator_model(input_dim)
    return discriminator

