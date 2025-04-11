# BertGCN-LDA Text Classification Project

This project combines **BERT** pretrained language models with **Graph Convolutional Networks (GCN)**, optionally integrating **LDA** topic features, for text classification tasks on benchmark datasets such as 20NG, R8, R52, Ohsumed, and MR.

---

## 🛠 Environment Setup

Recommended environment:

- Python >= 3.8
- PyTorch >= 1.10
- DGL >= 0.9.0
- Transformers
- Scikit-learn
- Gensim
- tqdm
- ignite

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 📂 Data Preparation

Ensure your data directory structure looks like this:
```
data/
  corpus/
    20ng_shuffle.txt
    20ng_vocab.txt
    ...
```

Main files required:
- `{dataset}_shuffle.txt`: Shuffled document texts
- `{dataset}_vocab.txt`: Vocabulary list
- Graph matrices (e.g., `adj`, `features`) generated using preprocessing scripts

---

## 🚀 Run Order

1. First, use the `build_graph.py` script to construct the graph structure.
2. Then select the appropriate training script based on the task type:
   - Single-label classification: Use `train_bert_gcn.py` or `train_bert_gcn_lda.py`
   - Multi-label classification: Use `train_bert_gcn_lda_muti.py`

---

## 🚀 Model Training

### 1. Train BERT + GCN
```bash
python train_bert_gcn.py \
  --dataset 20ng \
  --bert_init roberta-base \
  --gcn_model gcn \
  --batch_size 64 \
  --nb_epochs 50
```

### 2. Train BERT + GCN-LDA
```bash
python train_bert_gcn_lda.py \
  --dataset 20ng \
  --bert_init roberta-base \
  --gcn_model gcnLda \
  --num_topics 64 \
  --floats 0.3 0.5 0.2
```

### 3. Multi-label Classification Training
For **multi-label classification tasks**, use the `train_bert_gcn_lda_muti.py` script.

Example command:
```bash
python train_bert_gcn_lda_muti.py \
  --dataset your_dataset \
  --bert_init roberta-base \
  --gcn_model gcnLda \
  --num_topics 64 \
  --floats 0.3 0.5 0.2
```

**Dataset Preparation Notes:**
- Recommended public dataset: **Reuters Corpus**.
- Due to copyright issues, data files are not provided.
- You can also build your own dataset, e.g., by downloading the **Derwent patent dataset** from **Web of Science**.

### Main Parameter Description
- `--dataset`: Dataset name (e.g., 20ng, R8, R52, ohsumed, mr)
- `--bert_init`: Pretrained BERT model to use
- `--gcn_model`: Choose GCN or GCN-LDA
- `--num_topics`: Number of LDA topics (only for GCN-LDA)
- `--floats`: Multi-scale feature mixing ratios
- Other training control parameters like `--batch_size`, `--nb_epochs`, `--gcn_layers`

After training, logs and model checkpoints will be saved automatically under the `checkpoint/` directory.

---

## 📈 Output

After training you will get:
- Best model checkpoint at `checkpoint/{model}_{dataset}/checkpoint.pth`
- Training log file `training.log`
- Accuracy (Accuracy) and loss (Loss) recorded for each epoch

---

## 📋 Notes

- Single GPU training by default (you can manually specify GPU device)
- Supports mixed graph with document and word nodes
- Node features (`cls_feats`) are updated with the latest BERT embeddings every epoch
- Learning rate scheduling applied (decay after 30 epochs)

---

## ✨ Acknowledgements

This project is based on the following paper:

- Yuxiao Lin, Yuxian Meng, Xiaofei Sun, Qinghong Han, Kun Kuang, Jiwei Li, Fei Wu. "BertGCN: Transductive Text Classification by Combining GCN and BERT." Findings of ACL 2021. [Paper Link](https://arxiv.org/abs/2105.05727) [Code Link](https://github.com/ZeroRin/BertGCN)

---

Enjoy building graph-enhanced text classifiers! 🚀
