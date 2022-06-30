from torch.optim import ASGD, SGD, Adadelta, Adagrad, Adam, Adamax, RMSprop

key2opt = {
    "sgd": SGD,
    "adam": Adam,
    "asgd": ASGD,
    "adamax": Adamax,
    "adadelta": Adadelta,
    "adagrad": Adagrad,
    "rmsprop": RMSprop,
}


def get_optimizer(opt_name):
    return key2opt[opt_name]
