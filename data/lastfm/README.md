## LastFM Dataset

Originally from [HetRec 2011](https://grouplens.org/datasets/hetrec-2011/).

Preprocessed by [MAGNN](https://github.com/cynricfu/MAGNN).

We discard the negative edges provided by MAGNN and samples hard negative edges for validtaion and testing by ourselves.

### Usage

1. Create a folder named `magnn` in this directory.
2. Download `magnn.zip` from [this Dropbox link](https://www.dropbox.com/s/0er61udwe2msd2h/magnn.zip?dl=0).
3. Extract and copy everything into the `magnn` folder.
4. Run [preprocess_lastfm.ipynb](/preprocess_lastfm.ipynb) to preprocess this dataset.
