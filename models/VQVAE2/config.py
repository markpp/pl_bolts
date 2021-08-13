from argparse import Namespace

hparams = Namespace(**{# data
                       'name': 'VQVAE2',
                       'dataset': 'training',
                       'data': '../../../data/sewer/paper/',
                       'save_dir': 'logs/',
                       'image_size': 128,
                       # model
                       'in_channels':              3,
                       'hidden_channels':          128,
                       'res_channels':             32,
                       'nb_res_layers':            2,
                       'embed_dim':                64,
                       'nb_entries':               512,
                       'nb_levels':                3,
                       'scaling_rates':            [4, 2, 2],
                       # training
                       'gpus': 1,
                       'max_epochs': 100,
                       'learning_rate': 1e-4,
                       'beta': 0.25,
                       'beta1': 0.9,
                       'beta2': 0.999,
                       'batch_size': 128,
                       'n_workers':16})
