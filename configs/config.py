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

    'SIMULATION' : {
        'EPOCH' : 20,
        'topK' : 10

    },

    'USER_SIMULATOR' : {
        'DIM_EMB' : 5,
        'EPOCH' : 20,
        'OPTIMIZER' : {
            'name' : 'adagrad', 
            'lr': 0.05, 
            'weight_decay': 1e-4
        },
        'LOSS' : 'bce',
        'AUC_THRESH' : 0.74,
        'BS': {
            'train' : 8192,
            'val' : 4096,
            'test' : 4096
        },
        'DROPOUT' : 0.1
    }, 

    'INITIAL_RECOMMENDER' : {
        'DIM_EMB' : 5,
        'EPOCH' : 1,
        'OPTIMIZER' : {
            'name' : 'adagrad', 
            'lr': 0.05, 
            'weight_decay': 1e-4
        },
        'LOSS' : 'bce',
        'AUC_THRESH' : 0.70,
        'BS': {
            'train' : 8192,
            'val' : 4096,
            'test' : 4096
        },
        'DROPOUT' : 0.1
    },

    'REPLAY_BUFFER' : {
        'SIZE' : 1000000,
        'BS' : 4096,
    }, 

    'RL' : {
        'EPOCH' : 300,
    }, 


    'ACTOR_CRITIC' : {
        'GAMMA' : 0.995,
        'POLICY_LR' : 1e-3,
        'VALUE_LR' : 1e-3,
        'POLICY_HIDDEN_DIM' : 256,
        'VALUE_HIDDEN_DIM' : 256,
    }
}