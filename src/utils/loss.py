import torch.nn as nn

key2loss = {"bce": nn.BCEWithLogitsLoss()}


def get_loss_function(loss_name):
    return key2loss[loss_name]
