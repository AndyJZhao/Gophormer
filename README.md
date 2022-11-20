# Gophormer: Ego-Graph Transformer for Node Classification 	
This repository is an implementation of Gophormer - [Gophormer: Ego-Graph Transformer for Node Classification](https://arxiv.org/abs/2110.13094). The work is completed during Jianan's internship at Microsoft Research Asia.

## Requirements
We use the miniconda to manage the python environment. We provide the environment.yaml to implement the environment
```
conda env create -f environment.yml
```

## Running

Flickr
```
python src/Gophormer/models/Gophormer/train.py --batch_size=32 --cr_temperature=0.3 --cr_weight=1 --dataset=flickr --dropout=0.1 --early_stop=30 --epochs=200 --eval_freq=3 --global_nodes=4 --gt_n_layers=5 --lr=0.0002 --n_samples=3 --n_workers=1 --norm=PreLN --proximity_encoding=1i --readout=MLP_1_center --readout_dropout=0 --sample=EgoGraph-5_1 --sample_dev=cpu --scheduler=Cos5_0.2_0.5_1 --trainer=node --val_inference=SubGVote-5 --wandb_name=FlickrRough --gpus=7
```

Blog
```
python src/Gophormer/models/Gophormer/train.py --batch_size=32 --cr_temperature=0.8 --cr_weight=0.8 --dataset=blogcatalog --dropout=0.1 --early_stop=30 --epochs=200 --eval_freq=3 --global_nodes=5 --gt_n_layers=4 --lr=0.0001 --n_samples=4 --n_workers=1 --norm=PreLN --proximity_encoding=1i --readout=MLP_1_center --readout_dropout=0 --sample=EgoGraph-7_1 --sample_dev=cpu --scheduler=Cos5_0.2_0.25_1.25 --trainer=node --val_inference=SubGVote-5 --wandb_name=FinalSearch-Blog --gpus=0
```
Pubmed
```
python src/Gophormer/models/Gophormer/train.py --batch_size=32 --cr_weight=0.5 --dataset=pubmed --dropout=0.3 --early_stop=30 --epochs=200 --eval_freq=3 --global_nodes=1 --gt_n_layers=4 --lr=0.00015 --n_samples=2 --n_workers=1 --norm=PreLN --proximity_encoding=1i --readout=MLP_1_center --readout_dropout=0 --sample=EgoGraph-5_2 --sample_dev=cpu --scheduler=Cos5_0.2_0.5_1.25 --trainer=node --val_inference=SubGVote-5 --wandb_name=PubmedRough_0.2 --gpus=1
```
DBLP
```
python src/Gophormer/models/Gophormer/train.py --batch_size=32 --cr_temperature=0.5 --cr_weight=0.5 --dataset=dblp --dropout=0.1 --early_stop=30 --epochs=200 --eval_freq=3 --global_nodes=3 --gt_n_layers=4 --lr=0.0001 --n_samples=2 --n_workers=1 --norm=PreLN --proximity_encoding=2i --readout=MLP_1_center --readout_dropout=0 --sample=EgoGraph-5_2 --sample_dev=cpu --scheduler=Cos5_0.2_0.25_1.25 --trainer=node --val_inference=SubGVote-5 --wandb_name=DBLP_Rough_Bayes --gpus=0
```
Citeseer
```
python src/Gophormer/models/Gophormer/train.py --batch_size=32 --cr_temperature=0.4 --cr_weight=0.3 --dataset=citeseer --dropout=0.4 --early_stop=30 --epochs=200 --eval_freq=3 --global_nodes=2 --gt_n_layers=2 --lr=0.0002 --n_samples=4 --n_workers=1 --norm=PreLN --proximity_encoding=1i --readout=MLP_1_center --readout_dropout=0 --sample=EgoGraph-7_2 --sample_dev=cpu --scheduler=Cos5_0.2_0.25_1.25 --trainer=node --val_inference=SubGVote-5 --wandb_name=FinalSearch-Citeseer --gpus=3
```
Cora
```
python src/Gophormer/models/Gophormer/train.py --batch_size=32 --cr_weight=0.5 --dataset=cora --dropout=0.30000000000000004 --early_stop=30 --epochs=200 --eval_freq=3 --global_nodes=4 --gt_n_layers=2 --lr=0.00015000000000000001 --n_samples=5 --n_workers=1 --norm=PreLN --proximity_encoding=1i --readout=MLP_1_center --readout_dropout=0 --sample=EgoGraph-7_2 --sample_dev=cpu --scheduler=Cos5_0.2_0.5_1 --trainer=node --val_inference=SubGVote-5 --wandb_name=coraRough_0.3 --gpus=1
```


## Citation
If you find our work useful, please consider citing our work:
```

@article{DBLP:journals/corr/abs-2110-13094,
  author    = {Jianan Zhao and
               Chaozhuo Li and
               Qianlong Wen and
               Yiqi Wang and
               Yuming Liu and
               Hao Sun and
               Xing Xie and
               Yanfang Ye},
  title     = {Gophormer: Ego-Graph Transformer for Node Classification},
  journal   = {CoRR},
  volume    = {abs/2110.13094},
  year      = {2021},
  url       = {https://arxiv.org/abs/2110.13094},
  eprinttype = {arXiv},
  eprint    = {2110.13094},
  timestamp = {Tue, 18 Oct 2022 14:44:48 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2110-13094.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
}
```