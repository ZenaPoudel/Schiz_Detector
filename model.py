import torch
import torch.nn as nn
import torch.nn.functional as F


def model_3DCNN (dropout=0.3):
    '''
    Function defining baseline 3DCNN model based on the paper 
    "Oh Jihoon, Oh Baek-Lok, Lee Kyong-Uk, Chae Jeong-Ho, Yun Kyongsik, 
    Identifying Schizophrenia Using Structural MRI With a Deep Learning Algorithm, 
    Frontiers in Psychiatry, Volume 11, 2020, ISSN 1664-0640, 
    https://doi.org/10.3389/fpsyt.2020.00016" but changing the value of dropout and adding
    the softmax layer at the end.

    Arg: 
        dropout: float [ 0 to 1 ]  
    
    returns: model

    '''
    model = nn.Sequential(
        nn.Conv3d(1, 32, [3,3,3], stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None),
        nn.ReLU(inplace=False),
        nn.Conv3d(32, 32, [3,3,3], stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None),
        torch.nn.ReLU(inplace=False),
        nn.MaxPool3d([3,3,3], stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False),
        nn.Dropout(p=dropout, inplace=False),
        nn.Conv3d(32, 64, [3,3,3], stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None),
        nn.ReLU(inplace=False),
        nn.Conv3d(64, 64, [3,3,3], stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None),
        torch.nn.ReLU(inplace=False),
        nn.MaxPool3d([3,3,3], stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False),
        nn.Dropout(p=dropout, inplace=False),
        nn.Flatten(start_dim=1, end_dim=- 1),
        nn.Linear(46656, 512, bias=True, device=None, dtype=None),
        nn.Dropout(p=dropout, inplace=False),
        nn.Linear(512, 2, bias=True, device=None, dtype=None),
        nn.Softmax(dim=1)
    )

    if torch.cuda.is_available():
        model.cuda()
    
    return model