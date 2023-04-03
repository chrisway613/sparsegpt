import torch
import torch.nn as nn


DEV = torch.device('cuda:0')


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    """ 递归地在当前模块及其所有子模块中找到目标 layer """
    
    if type(module) in layers:
        return {name: module}
    
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers,
                name=f'{name}.{name1}' if name != '' else name1
            )
        )

    return res
