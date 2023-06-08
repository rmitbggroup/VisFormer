# VisFormer: Visualization Recommendation Through Visual Relation Learning and Visual Preference Learning

# Overview
data/: raw data and processed data. You need to download the raw dataset here.

data_preprocessing/: related python files for data preprocessing

model/: model definition, training scripts, and evaluation scripts

tokenizer/: downloaded files used in table serialization

requirements/: python dependencies

# Data Preprocessing
## Dataset Download
In this paper, we used Plotly Community Feed dataset. You can download the origin dataset from https://github.com/mitmedialab/vizml. 

## Data Preparing
The related files are included in `data_preprocessing` directory. 
To pre-process the dataset:
- run `python filter.py` to remove records that have missing values in visualization specification and table data. We also remove records whose chart type are not in four major type (scatter, line, bar, pie). The rest records are saved in `data/filtered_data.tsv`.
- run `python extract_info.py` to extract relevant table and visualization specification from the rest records, which are saved in the directory `data/processed_data/.` It also creates a file `data/dataset.tsv` to save the dataset information, such as the path of table and charts.
- run `python deduplicate.py` to deduplicate the (table, charts) pairs in `data/dataset.tsv` that are very similar.
- run `python split.py --ratio=para` to split the dataset information `data/dataset.tsv` into training dataset `data/training_data.tsv` and test dataset `data/test_data.tsv`, where `para` denotes the ratio of training data to the whole data.

## Data generator
We aslo need to create visualization images to train Visual Preference Learning Model. The file is also in 'data_preprocessing' directory.
- run `python img_generator.py` to generate corresponding visualization images for each table according to the corresponding visualization specifications. And the images are saved in 'data/image' directory.

# Model Training and Method Evaluation
The related file are included in `model` directory.

## Model Training
-- run `python train_vrl.py` and `python train_vpl.py` to train the Visual Relation Learning model and Visual Preference Learning model, respectively.

## Method Evaluation
-- run 'python evaluation.py' to evaluate the performance of VisFormer.
