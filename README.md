## MAGNN

This repository provides a reference implementation of MECCH as described in the paper:
> MECCH: Metapath Context Convolution-based Heterogeneous Graph Neural Networks.

TODO

### Dependencies

* PyTorch 1.10
* DGL 0.7
* scikit-learn
* tqdm

### Datasets

* IMDB
* ACM
* DBLP
* LastFM
* PubMed

TODO

### Usage

```
python main.py [-h] --model MODEL --dataset DATASET [--task TASK] [--gpu GPU] [--config CONFIG] [--repeat REPEAT]
```

```
optional arguments:
  -h, --help            show this help message and exit
  --model MODEL, -m MODEL
                        name of model
  --dataset DATASET, -d DATASET
                        name of dataset
  --task TASK, -t TASK  type of task
  --gpu GPU, -g GPU     which gpu to use, specify -1 to use CPU
  --config CONFIG, -c CONFIG
                        config file for model hyperparameters
  --repeat REPEAT, -r REPEAT
                        repeat the training and testing for N times
```

### Citing

TODO
