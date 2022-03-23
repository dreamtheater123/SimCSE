import scipy.stats
from torch.utils.data import DataLoader
from transformers import get_scheduler, AutoTokenizer, BertModel, BertConfig
from Config import args, device
import torch
import os
from utils import load_testset
from train import cos_1by1
from tqdm import tqdm

# STS12 predicting
input_filenames = ['STS.input.MSRpar.txt', 'STS.input.MSRvid.txt', 'STS.input.SMTeuroparl.txt',
                   'STS.input.surprise.OnWN.txt', 'STS.input.surprise.SMTnews.txt']
gs_filenames = [item.replace('input', 'gs') for item in input_filenames]

input_all, gs_all = [], []
p_all, h_all = [], []
for i_file, g_file in zip(input_filenames, gs_filenames):
    with open('./dataset/predict/STS12-en-test/' + i_file, 'r', encoding='utf-8') as f:
        i_lines = f.readlines()
    with open('./dataset/predict/STS12-en-test/' + g_file, 'r', encoding='utf-8') as f:
        g_lines = f.readlines()
    i_lines = [item.strip().split('\t') for item in i_lines]
    g_lines = [float(item.strip()) for item in g_lines]
    input_all.extend(i_lines)
    gs_all.extend(g_lines)
for item in input_all:
    p_all.append(item[0])
    h_all.append(item[1])
print(len(input_all), len(p_all), len(h_all), len(gs_all))

tokenizer = AutoTokenizer.from_pretrained("./model/bert-base-uncased")
model = BertModel.from_pretrained("./model/bert-base-uncased")
model.load_state_dict(torch.load(os.path.join('./log', args['test_model'][0], args['test_model'][1],
                                              'model/newest_model.ckpt'),  map_location=device), strict=True)

# load data
p_all, h_all = load_testset(args, (p_all, h_all), tokenizer)
data_loader_p = DataLoader(p_all, batch_size=args['batch_size'], shuffle=False)
data_loader_h = DataLoader(h_all, batch_size=args['batch_size'], shuffle=False)
# predict
pred_all = []
model.eval()
with torch.no_grad():
    for val_batch_p, val_batch_h in tqdm(zip(data_loader_p, data_loader_h)):
        val_batch_p = {k: v.to(device) for k, v in val_batch_p.items()}
        val_batch_h = {k: v.to(device) for k, v in val_batch_h.items()}
        val_outputs_p = model(**val_batch_p)
        val_outputs_h = model(**val_batch_h)
        val_cls_p = val_outputs_p.last_hidden_state[:, 0, :]  # 取BERT输出的CLS embedding shape=(batch, embedding)
        val_cls_h = val_outputs_h.last_hidden_state[:, 0, :]  # 取BERT输出的CLS embedding shape=(batch, embedding)
        pred = cos_1by1(val_cls_p, val_cls_h)
        pred = pred.squeeze().tolist()
        pred_all.extend(pred)
        # TODO: cosine, concat, spearman (spearman correlation takes the cos value and gold standard)

    print(scipy.stats.spearmanr(gs_all, pred_all).correlation)
    print(scipy.stats.spearmanr(pred_all, gs_all).correlation)
