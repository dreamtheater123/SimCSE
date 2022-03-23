import os
import pickle as pkl
import time
from time import gmtime, strftime
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from tqdm import tqdm

# import related to torch
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
from tensorboardX import SummaryWriter

# import related to transformers
from transformers import get_scheduler, AutoTokenizer, BertModel, BertConfig
from transformers import AdamW
from datasets import load_metric

from utils import load_hf_dataset
from Config import args, device


def l2norm(vec):
    return vec / (torch.norm(vec, dim=1, keepdim=True).expand_as(vec) + 1e-8)  # 1e-8 is to avoid dividing by 0


def mat_cos(p, h):
    """
    matrix cosine similarity
    :return:
    """
    p = l2norm(p)
    h = l2norm(h)
    cos = torch.mm(p, h.T)

    return cos


def cos_1by1(p, h):
    """
    calculate cosine similarity element-wize
    :param p:
    :param h:
    :return:
    """
    p = l2norm(p)
    h = l2norm(h)
    cos = torch.mul(p, h)
    cos = torch.sum(cos, dim=1, keepdim=True)

    return cos


def train_hf(args, bert_model, train_dataset, val_dataset):
    train_data_loader_p = DataLoader(train_dataset[0], batch_size=args['batch_size'], shuffle=False)
    train_data_loader_h = DataLoader(train_dataset[1], batch_size=args['batch_size'], shuffle=False)
    val_data_loader_p = DataLoader(val_dataset[0], batch_size=args['batch_size'], shuffle=False)
    val_data_loader_h = DataLoader(val_dataset[0], batch_size=args['batch_size'], shuffle=False)
    train_data_loader_n = DataLoader(train_dataset[2], batch_size=args['batch_size'], shuffle=False)
    val_data_loader_n = DataLoader(val_dataset[2], batch_size=args['batch_size'], shuffle=False)

    num_training_steps = args['epoch'] * len(train_data_loader_p)
    model = bert_model.to(device)
    optimizer = AdamW(model.parameters(), lr=args['learning_rate'])  # default: AdamW
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0,
                                 num_training_steps=num_training_steps)
    criterion = nn.CrossEntropyLoss()
    args['round2eval'] = min(args['round2eval'], len(train_data_loader_p))
    print('len of dataloader:', len(train_data_loader_p), ', round2eval:', args['round2eval'])

    # tensorboard
    if args['enable_writer']:
        writer_path = os.path.join('./log', args['model_time'], 'tensorboard')
        if not os.path.exists(writer_path):
            os.makedirs(writer_path)
        writer = SummaryWriter(log_dir=writer_path)

    # training
    model.train()
    num_batch = 0
    loss_train = 0
    min_dev_loss = 10000
    tik = time.time()
    for epoch in range(args['epoch']):
        print(f"Epoch [{epoch + 1:} / {args['epoch']:}]")
        for batch_p, batch_h, batch_n in tqdm(zip(train_data_loader_p, train_data_loader_h, train_data_loader_n)):
            model.train()
            num_batch += 1  # that makes the batch number of the first step 1
            # forward
            batch_p = {k: v.to(device) for k, v in batch_p.items()}
            batch_h = {k: v.to(device) for k, v in batch_h.items()}
            outputs_p = model(**batch_p)
            outputs_h = model(**batch_h)
            cls_p = outputs_p.last_hidden_state[:, 0, :]  # extract embedding of the CLS token in BERT
            cls_h = outputs_h.last_hidden_state[:, 0, :]  # extract embedding of the CLS token in BERT
            pos = cos_1by1(cls_p, cls_h)  # positive embedding
            neg = mat_cos(cls_p, cls_h)  # negative embedding
            mask = (torch.eye(neg.shape[0]) * 1e12).to(device)  # Identity matrix (for softmax masking)
            neg = neg - mask
            logits = torch.cat((pos, neg), dim=1)

            # if using hard negative
            if args['use_hard_neg']:
                batch_n = {k: v.to(device) for k, v in batch_n.items()}
                outputs_n = model(**batch_n)
                cls_n = outputs_n.last_hidden_state[:, 0, :]  # CLS embedding
                neg2 = mat_cos(cls_p, cls_n)  # negative embedding for hard negative samples
                logits = torch.cat((logits, neg2), dim=1)

            logits = logits / args['temp']
            loss_target = torch.zeros(logits.shape[0], dtype=torch.long).to(device)  # requires_grad = False
            # backward
            optimizer.zero_grad()
            batch_loss = criterion(logits, loss_target)
            loss_train += batch_loss.item()
            batch_loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # eval
            if num_batch % args['round2eval'] == 0 or num_batch % args['round2eval'] == len(train_data_loader_p):
                # eval for a certain of batches or a whole epoch
                model.eval()
                with torch.no_grad():
                    val_loss = 0
                    for val_batch_p, val_batch_h, val_batch_n in zip(val_data_loader_p, val_data_loader_h, val_data_loader_n):
                        val_batch_p = {k: v.to(device) for k, v in val_batch_p.items()}
                        val_batch_h = {k: v.to(device) for k, v in val_batch_h.items()}
                        val_outputs_p = model(**val_batch_p)
                        val_outputs_h = model(**val_batch_h)
                        val_cls_p = val_outputs_p.last_hidden_state[:, 0, :]  # CLS embedding
                        val_cls_h = val_outputs_h.last_hidden_state[:, 0, :]  # CLS embedding
                        val_pos = cos_1by1(val_cls_p, val_cls_h)  # positive embedding
                        val_neg = mat_cos(val_cls_p, val_cls_h)  # negative embedding
                        val_mask = (torch.eye(val_neg.shape[0]) * 1e12).to(device)  # identity matrix for masking
                        val_neg = val_neg - val_mask
                        val_logits = torch.cat((val_pos, val_neg), dim=1)

                        # if using hard negative
                        if args['use_hard_neg']:
                            val_batch_n = {k: v.to(device) for k, v in val_batch_n.items()}
                            val_outputs_n = model(**val_batch_n)
                            val_cls_n = val_outputs_n.last_hidden_state[:, 0, :]  # CLS embedding
                            val_neg2 = mat_cos(val_cls_p, val_cls_n)  # negative embedding for hard negative samples
                            val_logits = torch.cat((val_logits, val_neg2), dim=1)

                        val_logits = val_logits / args['temp']
                        val_loss_target = torch.zeros(val_logits.shape[0], dtype=torch.long).to(device)  # requires_grad = False
                        val_batch_loss = criterion(val_logits, val_loss_target)
                        val_loss += val_batch_loss.item()
                    
                    loss_train /= args['round2eval']
                    val_loss /= len(val_data_loader_p)
                    # save model
                    model_path = os.path.join('./log', args['model_time'], 'model')
                    if not os.path.exists(model_path):
                        os.makedirs(model_path)
                    torch.save(model.state_dict(), os.path.join(model_path, 'newest_model.ckpt'))

                    if val_loss < min_dev_loss:
                        min_dev_loss = val_loss
                        improve = '*'
                        # save best model
                        torch.save(model.state_dict(), os.path.join(model_path, 'best_model.ckpt'))
                    else:
                        improve = ' '

                    time_spent = (time.time() - tik) / 60.0
                    print(args['model_time'])
                    print('# batch:', num_batch, '|', 'training loss:', round(loss_train, 4), '|', 'val loss:',
                          round(val_loss, 4), '|', 'time:', round(time_spent, 2), '|', 'improve:', improve)  # standard output

                if args['enable_writer']:
                    writer.add_scalar('hyperparameters/lr', optimizer.state_dict()['param_groups'][0]['lr'], num_batch)
                    writer.add_scalar('loss/train', loss_train, num_batch)
                    writer.add_scalar('loss/dev', val_loss, num_batch)
                    # writer.add_scalar('acc/dev', val_acc, num_batch)
                    # writer.add_scalar('acc/test', test_acc, num_batch)
                    # writer.add_figure("Confusion matrix", cm_figure, num_batch)
                loss_train = 0  # since we use # of batch to perform validation & testing
    if args['enable_writer']:
        writer.close()


def main():
    args['model_time'] = strftime('%Y-%m-%d_%H_%M_%S', gmtime())
    # load dataset
    file_name = os.path.join('./dataset', args['which_dataset'])
    # file_name = './dataset/test_dataloader.csv'
    data = pd.read_csv(file_name)

    # load model
    tokenizer = AutoTokenizer.from_pretrained("./model/bert-base-uncased")
    bert_model = BertModel.from_pretrained("./model/bert-base-uncased")
    train_dataset, val_dataset = load_hf_dataset(args, data, tokenizer)

    # training
    print('training start!')
    train_hf(args, bert_model, train_dataset, val_dataset)
    print('training finished!')


if __name__ == '__main__':
    main()
