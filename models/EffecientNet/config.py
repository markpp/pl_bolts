
# Coefficients:   width,depth,res,dropout
#'efficientnet-b0': (1.0, 1.0, 224, 0.2),'efficientnet-b1': (1.0, 1.1, 240, 0.2),
#'efficientnet-b2': (1.1, 1.2, 260, 0.3),'efficientnet-b3': (1.2, 1.4, 300, 0.3),
#'efficientnet-b4': (1.4, 1.8, 380, 0.4),'efficientnet-b5': (1.6, 2.2, 456, 0.4),
#'efficientnet-b6': (1.8, 2.6, 528, 0.5),'efficientnet-b7': (2.0, 3.1, 600, 0.5),

from argparse import Namespace
hparams = Namespace(**{# data
                       'name': 'EffecientNet',
                       'dataset': 'mlp',
                       'data': '../../../data/sewer/paper/',
                       'save_dir': 'logs/',
                       'image_size': 224,
                       'in_channels': 3,
                       # model
                       'arch': 'efficientnet-b0',
                       'output_size': 1,
                       # training
                       'gpus': 1,
                       'max_epochs': 100,
                       'learning_rate': 1e-4,
                       'batch_size': 96,
                       'n_workers':16})
