# Fast and accurate interpretation of workload classification
These are codes for "Fast and accurate interpretation of workload classification", submitted to Plos One 2023.

## Dataset
We use a real-world workload dataset for classification.
The detailed information about dataset is summarized in the table below.

| dataset       | # of classes | input size | # of train | # of test of known |
|---------------|-------------:|-----------:|-----------:|-------------------:|
| SEC-seq       |           40 |       1735 |    586,885 |            293,444 |
| Memtest86-seq |           31 |       1599 |    433,334 |            216,696 |

We include the preprocessed data of Memtest86-seq dataset in this repository.
Due to the size limit, we upload `final_data` folder on google drive.
You can download the data from [[link]](https://drive.google.com/file/d/1JQQxuk3qUDAhCfNKZA3iBoYQxmeJOIWZ/view?usp=share_link).
`final_data` folder contains the entire feature vector for workload classification.
`sequences` folder includes n-gram sequences selected as CMD features.
The detailed structures of these directories are as follows:
```
current directory
├── final_data
│   ├── 7-grams
│   ├── 11-grams
│   ├── 15-grams
│   ├── bank_access_counts
│   ├── data_split_ids
│   ├── test_ngram_zeros.npy
│   └── row_col_address_access_counts
└── sequences
    ├── 7-grams.top25_osr.pkl
    ├── 11-grams.top25_osr.pkl
    └── 15-grams.top25_osr.pkl
```

## Model
We provide a pre-trained MLP model for workload classification in `models` directory.
* `model_mlp.pt`: a trained 2-layer MLP model

## Code Description
All codes in this repository are implemented based on Python 3.7.
This repository contains the code for INFO, INterpretable model For wOrkload classification.

* The information of codes implemented in this repository.
  * `main.py`: train and evaluate an interpretable model.
  * `cluster.py`: cluster features for workload classification to generate super features for interpretation.
  * `train.py`: train and evaluate a workload classification model.
  * `model.py`: implement MLP for classification.
  * `dataloader.py`: load workload sequence data and extract feature vectors for classification.

To interpret the classification results, you have to type the following command.
```shell
python main.py
```
You can train a workload classification model through the command below.
```shell
python train.py
```
These commands run the codes in the default setting.
