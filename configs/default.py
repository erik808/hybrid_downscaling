# ------------------------------------------------------------------
# Downscaling AutoEncoder hyper parameters and tuning configurations
# ------------------------------------------------------------------
import numpy as np

# default hyper parameters
hyper_params = {
    'history' : 'all',
    'future' : 400,
    'epochs' : 5,
    'batch_size' : 4,
    'unroll_dim' : 0,
    'lookback' : 0,
    'noise_stddev' : 0.0,
    'dropout_rate' : 0.0,
    'num_conv_blocks' : 1,
    'conv_layers_per_block' : 1,
    'num_feedthrough_filters' : 112,
    'num_feedthrough_layers' : 2,
    'num_output_layers' : 2,
    'optimizer' : 'adam',
    'L2_lambda' : 0.0,
    'kernel_size' : (3,3),
    'latent_space_model' : 'VAE',
    'latent_space_dim' : 4,
    'learning_rate' : 0.002,
    'num_filters' : 32,
    'num_filters_last' : 112,
    'downsample_stride' : (2,2),
}


# dict for use with optuna. keys represent different grid search or
# other parameter tuning experiments
tuning_config_dict = {}

__lookback_study__ = {
    'lookback' : {
        'type' : 'int',
        'args' : {
            'name' : 'lookback',
            'low'  : 0,
            'high' : 9,
        },
        'search_space' : np.arange(0,10).tolist(),
    },
}

tuning_config_dict['lookback_study'] = __lookback_study__


__everything__ = {
    'epochs' : {
        'type' : 'int',
        'args' : {
            'name': 'epochs',
            'low' : 1,
            'high': 50,
        },
        'search_space' : [10],
    },

    'learning_rate' : {
        'type' : 'float',
        'args' : {
            'name' : 'learning_rate',
            'low'  : 1e-4,
            'high' : 1e-2
        },
        'search_space' : [2e-3],
    },

    'num_conv_blocks' : {
        'type' : 'int',
        'args' : {
            'name' : 'num_conv_blocks',
            'low'  : 1,
            'high' : 6,
        },
        'search_space' : [1],
    },

    'conv_layers_per_block' : {
        'type' : 'int',
        'args' : {
            'name' : 'conv_layers_per_block',
            'low'  : 1,
            'high' : 6,
        },
        'search_space' : [1],
    },

    'num_filters' : {
        'type' : 'int',
        'args' : {
            'name' : 'num_filters',
            'low'  : 1,
            'high' : 200,
        },
        'search_space' : [64],
    },

    'num_filters_last' : {
        'type' : 'int',
        'args' : {
            'name' : 'num_filters_last',
            'low'  : 1,
            'high' : 100,
        },
        'search_space' : [112],
    },

    'num_feedthrough_layers' : {
        'type' : 'int',
        'args' : {
            'name' : 'num_feedthrough_layers',
            'low'  : 1,
            'high' : 100,
        },
        'search_space' : [2],
    },

    'num_feedthrough_filters' : {
        'type' : 'int',
        'args' : {
            'name' : 'num_feedthrough_filters',
            'low'  : 1,
            'high' : 1000,
        },
        'search_space' : [112],
    },

    'num_output_layers' : {
        'type' : 'int',
        'args' : {
            'name' : 'num_output_layers',
            'low'  : 1,
            'high' : 5,
        },
        'search_space' : [2],
    },

    'lookback' : {
        'type' : 'int',
        'args' : {
            'name' : 'lookback',
            'low'  : 0,
            'high' : 9,
        },
        'search_space' : [2],
    },

    'unroll_dim' : {
        'type' : 'int',
        'args' : {
            'name' : 'unroll_dim',
            'low'  : 0,
            'high' : 9,
        },
        'search_space' : [0],
    },
}



__latent_space_dim__ = {
    'latent_space_dim' : {
        'type' : 'int',
        'args' : {
            'name' : 'latent_space_dim',
            'low'  : 1,
            'high' : 10000,
        },
        'search_space' : [4],
    },
    'lookback' : {
        'type' : 'int',
        'args' : {
            'name' : 'lookback',
            'low'  : 0,
            'high' : 9,
        },
        'search_space' : [3],
    },
    'latent_space_model' : {
        'type'  : 'categorical',
        'args'  : {
            'name':'latent_space_model',
            'choices' : [
                'RNN',
                'RNN_res',
                'LSTM',
                'GRU',
                'ConvLSTM',
                'VAE',
                'VAE+RNN'
            ],
        },
        'search_space' : ['VAE+RNN'],
    },
}



__regularization__ = {
    'L2_lambda' : {
        'type' : 'float',
        'args' : {
            'name' : 'L2_lambda',
            'low'  : 0,
            'high' : 1e2,
        },
        'search_space' : [0, 1e-8, 1e-7, 1e-6, 1e-5],
    },
},


# combine into big dict
tuning_config_dict['everything'] = __everything__
tuning_config_dict['latent_space_dim'] = __latent_space_dim__
tuning_config_dict['regularization'] = __regularization__
