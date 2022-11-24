## MECCH

This repository provides a reference implementation of MECCH as described in the paper [arXiv:2211.12792](https://arxiv.org/abs/2211.12792):
> MECCH: Metapath Context Convolution-based Heterogeneous Graph Neural Networks.<br>
> Xinyu Fu, Irwin King

### Dependencies

* PyTorch 1.10
* DGL 0.7
* scikit-learn
* tqdm

### Datasets

* [IMDB](data/imdb-gtn/README.md)
* [ACM](data/acm-gtn/README.md)
* [DBLP](data/dblp-gtn/README.md)
* [LastFM](data/lastfm/README.md)
* [PubMed](data/pubmed/README.md)

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

Before running the script, you need to first download and extract the datasets into correct locations. Please refer to the respective dataset README [above](#datasets).

After data preparation, the code can be easily run. For example, to run MECCH on the IMDB dataset for node classification using GPU, use the following command:
```
python main.py -m MECCH -t node_classification -d imdb-gtn -g 0
```
To run MECCH on the LastFM dataset for link prediction using GPU, use the following command:
```
python main.py -m MECCH -t link_prediction -d lastfm -g 0
```

### Citing

If you find MECCH useful in your research, please cite the following paper:
```
@article{fu2022mecch,
  author    = {Xinyu Fu and Irwin King},
  title     = {MECCH: Metapath Context Convolution-based Heterogeneous Graph Neural Networks},
  journal   = {CoRR},
  volume    = {abs/2211.12792},
  year      = {2022}
}
```
