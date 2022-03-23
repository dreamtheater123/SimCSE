# SimCSE: A lightweight pytorch implementation

This is a lightweight implementation of SimCSE. You can easily customize the model based on this implementation, and this is more user-friendly to python beginners. (Only supervised SimCSE is supported in this version)

[Link to SimCSE paper](https://arxiv.org/abs/2104.08821v2)

## Dataset
The supervised SimCSE dataset is `./dataset/nli_for_simcse.csv`. Training supervised SimCSE with and without hard negatives are supported. You can specify this setting in `Config.py`.

## Backbone Model
In this version, we only support BERT as the backbone of SimCSE. In order to correctly load the model, [bert-base-uncased](https://huggingface.co/bert-base-uncased) model needs to be downloaded into `./model` folder.

## Training
First, parameters need to be correctly set in `Config.py`. This is an approach to replace the command line parameters. Explanation of the parameters can be found within the file.<br>
Next, you can do the training by simply calling `python train.py`

## Prediction & results
In this version, we only support STS12 task here. Prediction command: `python prediction.py`<br>
Besides, In order to show that this implementation is correct, here's the results comparison of the STS12 task (we train the official SimCSE with the same configuration and test it on STS12 task):
| Implementation       | Spearman's correlation |
| -------------------  | ---------------------  |
| SimCSE official      | 0.761                  |
| Our implementation   | 0.763                  |
<br>
Hyperparameters: 
```
epoch: 1
batch_size: 32
temperature: 0.05
pooler: CLS
truncate: 32
```

## Features in the following versions
1. Unsupervised SimCSE.
2. Training along with the MLM objective.
3. Other STS tasks (and perhaps some transfer tasks).
4. Chinese version of SimCSE by pytorch.


### Have fun running (or customizing) the code!
