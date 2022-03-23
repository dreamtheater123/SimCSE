import torch

# configurations
args = {
    'which_dataset': 'nli_for_simcse_trial.csv',  # the dataset to be trained on
    'train_eval_split': 0.8,  # len(training_set) = len(dataset) * train_eval_split
    'batch_size': 64,
    'truncate': 32,  # truncate all sentences into the same length
    'use_hard_neg': False,  # whether to use hard negative to train supervised SimCSE
    'round2eval': 1000,  # evaluate the model after certain rounds of mini-batch training
    'enable_writer': False,  # wheter to enable tensorboard output
    'temp': 0.05,  # the temperture of InfoNCE loss
    'epoch': 10,
    'learning_rate': 5e-5,
    'test_model': ['BERT', 'no_neg_batch_32'],  # the path for loading model weight during predicting (you can change it yourself)
}

device_id = 0  # specify the GPU you want to use for running the experiments (only single-gpu training is supported in this version)
if torch.cuda.is_available():
    torch.torch.cuda.set_device(device_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    print(5e-5)
