import torch
import torch.nn as nn

softmax = nn.Softmax(dim=1)
sigmoid = nn.Sigmoid()


def gumbel_softmax(input_tensor, tau):

    positive_input = sigmoid(input_tensor)
    eps = 1e-10

    n_class = input_tensor.shape[1]

    g = -torch.log(-torch.log(torch.rand(n_class) + eps) + eps).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    z = softmax((torch.log(positive_input + eps) + g) / tau)

    return z
