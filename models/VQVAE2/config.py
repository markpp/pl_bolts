from argparse import Namespace

hparams = Namespace(**{# data
                       'name': 'VQVAE2',
                       'dataset': 'training',
                       'data': '../../../data/sewer/paper/',
                       'log_dir': 'logs/',
                       'image_shape': [3, 128, 128],
                       # model
                       'hidden_channels':          128,
                       'res_channels':             32,
                       'nb_res_layers':            2,
                       'embed_dim':                64,
                       'nb_entries':               512,
                       'nb_levels':                2,
                       'scaling_rates':            [4, 2], # first downscaled by factor of 4, then 2
                       # training
                       'gpus': 1,
                       'max_epochs': 20,
                       'learning_rate': 1e-4,
                       'beta': 0.25, # loss weighting
                       'batch_size': 128,
                       'n_workers':16})
