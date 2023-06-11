https://github.com/riccardomusmeci/lightning-ssl


python3 train_ssl.py --config config/ssl/dino_tiny.yml --model dino --data-dir /home/markpp/datasets/WeedSeason --checkpoint-dir /home/markpp/github/lightning-ssl/checkpoints

python3 linear_eval.py --ssl-config config/ssl/dino_tiny.yml --linear-config config/linear/config.yml --ssl-ckpt /home/markpp/github/lightning-ssl/checkpoints/epoch=180-step=27511-val_loss=4.682.ckpt --data-dir /home/markpp/datasets/WeedSeason --checkpoint-dir /home/markpp/github/lightning-ssl/checkpoints_linear

python3 extract_features.py --ssl-config config/ssl/dino_tiny.yml --linear-config config/linear/config.yml --ssl-ckpt /home/markpp/github/lightning-ssl/checkpoints/epoch=180-step=27511-val_loss=4.682.ckpt --data-dir /home/markpp/datasets/WeedSeason


    imgs = torch.stack([torch.tensor(x[0]) for x in batch], dim=0)
  File "/home/markpp/github/pl_bolts/models/DINO/src/utils.py", line 30, in <listcomp>
    imgs = torch.stack([torch.tensor(x[0]) for x in batch], dim=0)
ValueError: expected sequence of length 3 at dim 1 (got 6)
