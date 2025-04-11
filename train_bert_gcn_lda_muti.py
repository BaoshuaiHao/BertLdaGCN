import os
import sys
import csv
import pickle
import argparse
import shutil
import logging
from datetime import datetime

import torch as th
import torch.nn.functional as F
import torch.utils.data as Data
import dgl
import scipy.sparse as sp
from tqdm import tqdm

from gensim import corpora
from gensim.models import LdaModel
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss
from transformers import AutoTokenizer

from model import BertGCN, BertGAT, BertGCNLda
from utils import load_corpus, normalize_adj
from lda_process import get_topic_distribution

# ------------------------ Argument Parsing ------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--max_length', type=int, default=128, help='Input length for BERT')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('-m', '--m', type=float, default=0.7, help='Factor balancing BERT and GCN prediction')
parser.add_argument('--nb_epochs', type=int, default=50)
parser.add_argument('--bert_init', type=str, default='roberta-base',
                    choices=['roberta-base', 'roberta-large', 'bert-base-uncased', 'bert-large-uncased'])
parser.add_argument('--pretrained_bert_ckpt', default=None)
parser.add_argument('--dataset', default='20ng', choices=['20ng', 'R8', 'R52', 'ohsumed', 'mr'])
parser.add_argument('--checkpoint_dir', default=None, help='Checkpoint directory')
parser.add_argument('--gcn_model', type=str, default='gcn', choices=['gcn', 'gat', 'gcnLda'])
parser.add_argument('--gcn_layers', type=int, default=2)
parser.add_argument('--n_hidden', type=int, default=200, help='GCN hidden layer dimension')
parser.add_argument('--heads', type=int, default=8, help='Number of attention heads for GAT')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--gcn_lr', type=float, default=1e-3)
parser.add_argument('--bert_lr', type=float, default=1e-5)
parser.add_argument('--train_label', type=str, default="muti")
parser.add_argument('--num_topics', type=int, choices=[20,32,64,128,256,384,512], default=64)
parser.add_argument('--floats', type=float, nargs='+', help='Multi-scale mixing ratios')
args = parser.parse_args()

# ------------------------ Device Setup ------------------------
device = th.device('cuda' if th.cuda.is_available() else 'cpu')

# ------------------------ Logging Setup ------------------------
ckpt_dir = args.checkpoint_dir or f'./checkpoint/{args.bert_init}_{args.gcn_model}_{args.dataset}_{args.train_label}_{args.num_topics}'
os.makedirs(ckpt_dir, exist_ok=True)
shutil.copy(os.path.basename(__file__), ckpt_dir)

logger = logging.getLogger('training logger')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(formatter)
logger.addHandler(sh)

fh = logging.FileHandler(filename=os.path.join(ckpt_dir, 'training.log'), mode='w')
fh.setFormatter(formatter)
logger.addHandler(fh)

logger.info('Arguments:')
logger.info(str(args))
logger.info(f'Checkpoints will be saved in {ckpt_dir}')

# ------------------------ Load Data ------------------------
with open(f'data/corpus/{args.dataset}_shuffle.txt', 'r') as f:
    doc_content_list = [line.strip() for line in f]

# Load or generate LDA distributions
pkl_file = f'output/{args.num_topics}/distri.pkl'
if os.path.exists(pkl_file):
    with open(pkl_file, 'rb') as f:
        distributions = pickle.load(f)
else:
    dictionary = corpora.Dictionary.load(f'output/{args.num_topics}/dictionary.dict')
    lda_model = LdaModel.load(f'output/{args.num_topics}/lda.model')
    distributions = get_topic_distribution(doc_content_list, dictionary, lda_model)
    with open(pkl_file, 'wb') as f:
        pickle.dump(distributions, f)

lda_matrix_tensor = th.tensor(distributions, dtype=th.float32)

# Load Graph Corpus
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(args.dataset)

nb_node = features.shape[0]
nb_word = nb_node - train_size - test_size - val_mask.sum().item()
nb_class = y_train.shape[1]

# ------------------------ Model Initialization ------------------------
model = BertGCNLda(
    nb_class=nb_class,
    pretrained_model=args.bert_init,
    num_topics=args.num_topics,
    floats=args.floats,
    m=args.m,
    gcn_layers=args.gcn_layers,
    n_hidden=args.n_hidden,
    dropout=args.dropout
)

if args.pretrained_bert_ckpt:
    ckpt = th.load(args.pretrained_bert_ckpt, map_location=device)
    model.bert_model.load_state_dict(ckpt['bert_model'])
    model.classifier.load_state_dict(ckpt['classifier'])

# ------------------------ Input Encoding ------------------------
corpus_file = f'data/corpus/{args.dataset}_shuffle.txt'
with open(corpus_file, 'r') as f:
    texts = [line.replace('\\', '').strip() for line in f if line.strip()]

tokenizer = AutoTokenizer.from_pretrained(args.bert_init)
inputs = tokenizer(texts, max_length=args.max_length, truncation=True, padding='max_length', return_tensors='pt')

input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']

input_ids = th.cat([input_ids[:-test_size], th.zeros((nb_word, args.max_length), dtype=th.long), input_ids[-test_size:]])
attention_mask = th.cat([attention_mask[:-test_size], th.zeros((nb_word, args.max_length), dtype=th.long), attention_mask[-test_size:]])
lda_matrix_tensor = th.cat([lda_matrix_tensor[:-test_size], th.zeros((nb_word, args.num_topics)), lda_matrix_tensor[-test_size:]])

# ------------------------ Graph Construction ------------------------
adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
g = dgl.from_scipy(adj_norm.astype('float32'), eweight_name='edge_weight')

y = (y_train + y_val + y_test).argmax(axis=1)
y_train = y_train.argmax(axis=1)

g.ndata.update({
    'input_ids': input_ids,
    'attention_mask': attention_mask,
    'lda_matrix_tensor': lda_matrix_tensor,
    'label': th.LongTensor(y),
    'label_train': th.LongTensor(y_train),
    'train': th.FloatTensor(train_mask),
    'val': th.FloatTensor(val_mask),
    'test': th.FloatTensor(test_mask),
    'cls_feats': th.zeros((nb_node, model.feat_dim))
})

doc_mask = train_mask + val_mask + test_mask

logger.info('Graph Information:')
logger.info(str(g))

# ------------------------ Feature Updater ------------------------
def update_feature():
    model.eval()
    g_cpu = g.to('cpu')
    dataloader = Data.DataLoader(Data.TensorDataset(
        g_cpu.ndata['input_ids'][doc_mask],
        g_cpu.ndata['attention_mask'][doc_mask],
        g_cpu.ndata['lda_matrix_tensor'][doc_mask]
    ), batch_size=1024)

    cls_list = []
    with th.no_grad():
        for batch in tqdm(dataloader, desc='Updating Features'):
            input_ids, attention_mask, lda_feats = [x.to(device) for x in batch]
            output = model.bert_model(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0]
            cls_list.append(output.cpu())
    g.ndata['cls_feats'][doc_mask] = th.cat(cls_list, dim=0)

# ------------------------ Trainer and Evaluator ------------------------
optimizer = th.optim.Adam([
    {'params': model.bert_model.parameters(), 'lr': args.bert_lr},
    {'params': model.classifier.parameters(), 'lr': args.bert_lr},
    {'params': model.gcn.parameters(), 'lr': args.gcn_lr}
], lr=1e-3)

scheduler = th.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)

# Training Step
def train_step(engine, batch):
    model.train()
    optimizer.zero_grad()
    (idx,) = [x.to(device) for x in batch]
    train_mask = g.ndata['train'][idx].bool()
    y_pred = model(g.to(device), idx)[train_mask]
    y_true = g.ndata['label_train'][idx][train_mask]
    loss = F.nll_loss(y_pred, y_true)
    loss.backward()
    optimizer.step()
    return loss.item(), accuracy_score(y_true.cpu(), y_pred.argmax(dim=1).cpu())

trainer = Engine(train_step)

@trainer.on(Events.EPOCH_COMPLETED)
def reset_graph(trainer):
    scheduler.step()
    update_feature()
    th.cuda.empty_cache()

# Evaluation Step
def test_step(engine, batch):
    with th.no_grad():
        model.eval()
        (idx,) = [x.to(device) for x in batch]
        y_pred = model(g.to(device), idx)
        y_true = g.ndata['label'][idx]
        return y_pred, y_true

evaluator = Engine(test_step)
metrics = {'acc': Accuracy(), 'nll': Loss(F.nll_loss)}
for name, metric in metrics.items():
    metric.attach(evaluator, name)

# Evaluation Logging
@trainer.on(Events.EPOCH_COMPLETED)
def log_results(trainer):
    evaluator.run(Data.DataLoader(Data.TensorDataset(th.arange(nb_node)), batch_size=args.batch_size))
    metrics = evaluator.state.metrics
    logger.info(f"Epoch {trainer.state.epoch}: Train Acc: {metrics['acc']:.4f}, Loss: {metrics['nll']:.4f}")

log_results.best_val_acc = 0
update_feature()
trainer.run(Data.DataLoader(Data.TensorDataset(th.arange(nb_node)), batch_size=args.batch_size), max_epochs=args.nb_epochs)