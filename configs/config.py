CFG_DICT = {

    'DATASET' : {
        'PATH' : './datasets/ml-1m',
        
        'NUM_ITEMS' : 3883,
        'NUM_YEARS' : 81,
        'NUM_GENRES' : 18,

        'NUM_ZIPS' : 3439,
        'NUM_USERS' : 6040,
        'NUM_OCCUPS' : 21,
        'NUM_AGES' : 7,
        'NUM_SEX' : 2

    }, 
    'USER_SIMULATOR' : {
        'DIM_EMB' : 50,
        'EPOCH' : 20,
        'OPTIMIZER' : {
            'name' : 'adagrad', 
            'lr': 0.03, 
            'weight_decay': 1e-4
        },
        'LOSS' : 'bce',
        'AUC_THRESH' : 0.75,
        'BS': {
            'train' : 8192,
            'val' : 4096,
            'test' : 4096
        }
    }, 

    'INITIAL_RECOMMENDER' : {
        'DIM_EMB' : 10,
        'EPOCH' : 10,
        'OPTIMIZER' : {
            'name' : 'adagrad', 
            'lr': 0.03, 
            'weight_decay': 1e-4
        },
        'LOSS' : 'bce',
        'AUC_THRESH' : 0.70,
        'BS': {
            'train' : 8192,
            'val' : 4096,
            'test' : 4096
        }
    }
}