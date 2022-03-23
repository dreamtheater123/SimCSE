import json
import pickle
import matplotlib.pyplot as plt
import numpy as np
from Config import device
from tqdm import tqdm
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, BertModel
from datasets import Dataset, ClassLabel, Features, Value


def convert2dataset(args, dataset, tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=args['truncate'])
    data_p, data_h, data_n = [], [], []

    for line in tqdm(dataset.iterrows()):
        line = list(line[1])
        data_p.append(line[0])  # premise
        data_h.append(line[1])  # hypothesis
        data_n.append(line[2])  # negative
    print(len(data_p), len(data_h), len(data_n))

    # HF dataset settings
    text = Value(dtype='string')
    feature_dict = {'text': text}
    features = Features(feature_dict)
    p_data_dict = {'text': data_p}
    h_data_dict = {'text': data_h}
    n_data_dict = {'text': data_n}

    # create hf dataset
    hf_dataset_p = Dataset.from_dict(mapping=p_data_dict, features=features)
    hf_dataset_h = Dataset.from_dict(mapping=h_data_dict, features=features)
    hf_dataset_n = Dataset.from_dict(mapping=n_data_dict, features=features)
    hf_dataset_p = hf_dataset_p.map(tokenize_function, batched=True)
    hf_dataset_h = hf_dataset_h.map(tokenize_function, batched=True)
    hf_dataset_n = hf_dataset_n.map(tokenize_function, batched=True)
    hf_dataset_p = hf_dataset_p.remove_columns(["text"])
    hf_dataset_h = hf_dataset_h.remove_columns(["text"])
    hf_dataset_n = hf_dataset_n.remove_columns(["text"])
    hf_dataset_p.set_format("torch")
    hf_dataset_h.set_format("torch")
    hf_dataset_n.set_format("torch")

    return hf_dataset_p, hf_dataset_h, hf_dataset_n


def load_hf_dataset(args, data, tokenizer):
    split_index = int(len(data) * args['train_eval_split'])
    training_set = data[:split_index]
    val_set = data[split_index:]

    train_dataset = convert2dataset(args, training_set, tokenizer)
    val_dataset = convert2dataset(args, val_set, tokenizer)

    return train_dataset, val_dataset


def load_testset(args, dataset, tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=args['truncate'])
    data_p = dataset[0]
    data_h = dataset[1]

    # HF dataset settings
    text = Value(dtype='string')
    feature_dict = {'text': text}
    features = Features(feature_dict)
    p_data_dict = {'text': data_p}
    h_data_dict = {'text': data_h}

    # create hf dataset
    hf_dataset_p = Dataset.from_dict(mapping=p_data_dict, features=features)
    hf_dataset_h = Dataset.from_dict(mapping=h_data_dict, features=features)
    hf_dataset_p = hf_dataset_p.map(tokenize_function, batched=True)
    hf_dataset_h = hf_dataset_h.map(tokenize_function, batched=True)
    hf_dataset_p = hf_dataset_p.remove_columns(["text"])
    hf_dataset_h = hf_dataset_h.remove_columns(["text"])
    hf_dataset_p.set_format("torch")
    hf_dataset_h.set_format("torch")

    return hf_dataset_p, hf_dataset_h


if __name__ == '__main__':
    from Config import args
    import pandas as pd

    file_name = os.path.join('./dataset', args['which_dataset'])
    data = pd.read_csv(file_name)
    tokenizer = AutoTokenizer.from_pretrained("./model/bert-base-uncased")
    bert_model = BertModel.from_pretrained("./model/bert-base-uncased")
    print('successfully load model!')
    load_hf_dataset(args, data, tokenizer)
