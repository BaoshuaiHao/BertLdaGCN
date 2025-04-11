import torch as th
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, XLNetModel
from .torch_gcn import GCN
from .torch_gat import GAT
from torch import nn
import torch

"""多头注意力机制"""
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"

        # 定义用于计算Q、K、V的线性层
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        # 输出线性层
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        batch_size = x.size(0)
        seq_length = x.size(1)

        # 计算Q、K、V矩阵
        Q = self.query(x)  # (batch_size, seq_length, hidden_size)
        K = self.key(x)  # (batch_size, seq_length, hidden_size)
        V = self.value(x)  # (batch_size, seq_length, hidden_size)

        # 将Q、K、V矩阵分割成多个头
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32))
        attention_weights = torch.softmax(attention_scores, dim=-1)  # (batch_size, num_heads, seq_length, seq_length)

        # 使用注意力权重对V进行加权求和
        attention_output = torch.matmul(attention_weights, V)  # (batch_size, num_heads, seq_length, head_dim)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)

        # 通过最终的线性层生成输出
        output = self.out(attention_output)  # (batch_size, seq_length, hidden_size)

        return output, attention_weights


class LSTMWithAttention(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_layers=2, bidirectional=False, dropout=0.2, num_class=20):
        super().__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional,
                            batch_first=True)
        # self.attention = Attention(hidden_dim * 2 if bidirectional else hidden_dim)
        self.classifier = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, num_class)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        packed_output, (hidden, cell) = self.lstm(inputs)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        attn_output = self.attention(lstm_output)
        output = self.dropout(attn_output)
        output = self.Classifier(output)
        return output
class BertClassifier(th.nn.Module):
    def __init__(self, pretrained_model='', nb_class=20):
        super(BertClassifier, self).__init__()
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)

    def forward(self, input_ids, attention_mask):
        cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
        cls_logit = self.classifier(cls_feats)
        return cls_logit
class BertAVGClassifier(th.nn.Module):
    def __init__(self, pretrained_model='', nb_class=20):
        super(BertAVGClassifier, self).__init__()
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)

    def forward(self, input_ids, attention_mask):
        last_hidden_states = self.bert_model(input_ids, attention_mask).last_hidden_state
        cls_feats = last_hidden_states.mean(dim=1)
        cls_logit = self.classifier(cls_feats)
        return cls_logit
class AttBertClassifier(th.nn.Module):
    def __init__(self, pretrained_model='', nb_class=20, nb_head=2):
        super(AttBertClassifier, self).__init__()
        self.nb_class = nb_class
        self.nb_head = nb_head
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)
        self.hidden_size = self.bert_model.config.hidden_size
        self.muti_attention = MultiHeadSelfAttention(self.hidden_size, self.nb_head)

    def forward(self, input_ids, attention_mask):
        bert_feats = self.bert_model(input_ids, attention_mask).last_hidden_state
        output, attention_weights = self.muti_attention(bert_feats)
        cls_feats = output[:,0,:]
        cls_logit = self.classifier(cls_feats)
        return cls_logit
class LdaBertClassifier(th.nn.Module):
    def __init__(self, pretrained_model='', num_topics=128, nb_class=20):
        super(LdaBertClassifier, self).__init__()
        self.nb_class = nb_class
        self.num_topics = num_topics
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)
        self.lda_linear = th.nn.Linear(num_topics, nb_class)
        self.linear = nn.Linear(self.feat_dim+self.num_topics, nb_class)

    def forward(self, input_ids, attention_mask, matrix):
        cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
        cls_logit = self.classifier(cls_feats)
        cls_pred = th.nn.Softmax(dim=1)(cls_logit)
        lda_logit = self.lda_linear(matrix)
        lda_pred = th.nn.Softmax(dim=1)(lda_logit)
        pred = cls_pred * 0.8 + lda_pred * 0.2
        pred = th.log(pred)
        return pred
class LdaAttClassifier(th.nn.Module):
    def __init__(self, num_topics=128, nb_class=20, nb_head=8):
        super(LdaAttClassifier, self).__init__()
        self.nb_class = nb_class
        self.nb_head = nb_head
        self.num_topics = num_topics
        self.hidden_size = self.num_topics
        self.classifier = th.nn.Linear(self.num_topics, nb_class)
        self.muti_attention = MultiHeadSelfAttention(self.hidden_size, self.nb_head)
    def forward(self, matrix):
        feats = matrix.unsqueeze(0)
        output, attention_weights = self.muti_attention(feats)
        output = output.squeeze(0)
        cls_logit = self.classifier(output)
        cls_pred = th.nn.Softmax(dim=1)(cls_logit)
        pred = th.log(cls_pred)
        return pred
class LdaClassifier(th.nn.Module):
    def __init__(self, num_topics=128, nb_class=20):
        super(LdaClassifier, self).__init__()
        self.nb_class = nb_class
        self.num_topics = num_topics
        self.classifier = th.nn.Linear(self.num_topics, nb_class)

    def forward(self, matrix):
        cls_logit = self.classifier(matrix)
        cls_pred = th.nn.Softmax(dim=1)(cls_logit)
        pred = th.log(cls_pred)
        return pred

class LdaLSTMClassifier(th.nn.Module):
    def __init__(self, num_topics=128, nb_class=20):
        super(LdaLSTMClassifier, self).__init__()
        self.nb_class = nb_class
        self.embedding_dim = num_topics
        self.hidden_size = self.embedding_dim
        self.lstm = th.nn.LSTM(self.embedding_dim, self.hidden_size, batch_first=True,num_layers=2)
        self.bidirectional = True
        self.num_topics = num_topics
        self.classifier = th.nn.Linear(self.num_topics, nb_class)

    def forward(self, matrix):
        hid_out_lstm, (h_n, c_n) = self.lstm(matrix)
        cls_logit = self.classifier(hid_out_lstm)
        cls_pred = th.nn.Softmax(dim=1)(cls_logit)
        pred = th.log(cls_pred)
        return pred
class LdaBertLstmClassifier(th.nn.Module):
    def __init__(self, pretrained_model='', num_topics=128, nb_class=20, bidirectional=True, num_layers=2, embedding_dim = 768,hidden_dim = 768):
        super(LdaBertLstmClassifier, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional,batch_first=True)
        # self.attention = Attention(hidden_dim)
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)
        self.lda_liner = th.nn.Linear(num_topics, nb_class)
        self.lstm_liner = th.nn.Linear(768, nb_class)

    def forward(self, input_ids, attention_mask, matrix):
        sequence_lengths = attention_mask.sum(dim=1)
        bert_output = self.bert_model(input_ids, attention_mask)
        cls_feats = bert_output.pooler_output
        embedding = bert_output.last_hidden_state
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedding, sequence_lengths.cpu(), batch_first=True,
                                                            enforce_sorted=False)
        hid_out_lstm, (h_n, c_n) = self.lstm(packed_embedded)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(hid_out_lstm, batch_first=True)

        context_vector, attention_weights = self.attention(lstm_output)

        hidden = lstm_output[:,-1,:]

        cls_class = self.classifier(cls_feats)
        lstm_class = self.lstm_liner(context_vector)
        lda_output = self.lda_liner(matrix)
        cls_logit = cls_class + lda_output + lstm_class
        return cls_logit

class BertLstmClassifier(th.nn.Module):
    def __init__(self, pretrained_model='', nb_class=20, bidirectional=True, num_layers=16, embedding_dim=768,hidden_dim=768):
        super(BertLstmClassifier, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional,batch_first=True)
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(768, nb_class)

    def forward(self, input_ids, attention_mask):
        sequence_lengths = attention_mask.sum(dim=1)
        bert_output = self.bert_model(input_ids, attention_mask)
        embedding = bert_output.last_hidden_state
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedding, sequence_lengths.cpu(), batch_first=True,
                                                            enforce_sorted=False)
        hid_out_lstm, (h_n, c_n) = self.lstm(packed_embedded)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(hid_out_lstm, batch_first=True)
        hidden = lstm_output[:,-1,:]
        lstm_class = self.classifier(hidden)
        cls_logit = lstm_class
        cls_logit = th.nn.Softmax(dim=1)(cls_logit)
        return cls_logit

class BertLstmAttClassifier(th.nn.Module):
    def __init__(self, pretrained_model='', nb_class=20, bidirectional=True, nb_head=2, num_layers=2, embedding_dim = 768,hidden_dim = 768):
        super(BertLstmAttClassifier, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional,batch_first=True)
        self.nb_class = nb_class
        self.nb_head = nb_head
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.hidden_size = self.bert_model.config.hidden_size
        self.muti_attention = MultiHeadSelfAttention(self.hidden_size, self.nb_head)
        self.classifier = th.nn.Linear(768, nb_class)

    def forward(self, input_ids, attention_mask):
        sequence_lengths = attention_mask.sum(dim=1)
        bert_output = self.bert_model(input_ids, attention_mask)
        embedding = bert_output.last_hidden_state
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedding, sequence_lengths.cpu(), batch_first=True,
                                                            enforce_sorted=False)
        hid_out_lstm, (h_n, c_n) = self.lstm(packed_embedded)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(hid_out_lstm, batch_first=True)
        output, attention_weights = self.muti_attention(lstm_output)
        lstm_class = self.classifier(output[:,-1,:])
        cls_logit = lstm_class
        return cls_logit

class XlnetClassifier(th.nn.Module):
    def __init__(self, pretrained_model='', nb_class=20):
        super(XlnetClassifier, self).__init__()
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.xlnet_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = self.xlnet_model.d_model
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)

    def forward(self, input_ids, attention_mask):
        cls_feats = self.xlnet_model(input_ids, attention_mask).last_hidden_state
        output = cls_feats[:, -1, :].squeeze(1)
        cls_logit = self.classifier(output)
        return cls_logit

class XlnetGCN(th.nn.Module):
    def __init__(self, pretrained_model='roberta_base', nb_class=20, m=0.7, gcn_layers=2, n_hidden=200, dropout=0.5):
        super(XlnetGCN, self).__init__()
        self.m = m
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.xlnet_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = self.xlnet_model.d_model
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)
        self.gcn = GCN(
            in_feats=self.feat_dim,
            n_hidden=n_hidden,
            n_classes=nb_class,
            n_layers=gcn_layers-1,
            activation=F.elu,
            dropout=dropout
        )
    def forward(self, g, idx):
        input_ids, attention_mask = g.ndata['input_ids'][idx], g.ndata['attention_mask'][idx]
        if self.training:           # model.train()传过来的参数
            cls_feats = self.xlnet_model(input_ids, attention_mask)[0][:, 0]
            g.ndata['cls_feats'][idx] = cls_feats
        else:
            cls_feats = g.ndata['cls_feats'][idx]
        cls_logit = self.classifier(cls_feats)
        cls_pred = th.nn.Softmax(dim=1)(cls_logit)
        gcn_logit = self.gcn(g.ndata['cls_feats'], g, g.edata['edge_weight'])[idx]
        gcn_pred = th.nn.Softmax(dim=1)(gcn_logit)
        pred = (gcn_pred+1e-10) * self.m + cls_pred * (1 - self.m)
        pred = th.log(pred)
        return pred
class BertGCN(th.nn.Module):
    def __init__(self, pretrained_model='roberta_base', nb_class=20, m=0.7, gcn_layers=2, n_hidden=200, dropout=0.5):
        super(BertGCN, self).__init__()
        self.m = m
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)
        self.gcn = GCN(
            in_feats=self.feat_dim,
            n_hidden=n_hidden,
            n_classes=nb_class,
            n_layers=gcn_layers-1,
            activation=F.elu,
            dropout=dropout
        )

    def forward(self, g, idx):
        input_ids, attention_mask = g.ndata['input_ids'][idx], g.ndata['attention_mask'][idx]
        if self.training:           # model.train()传过来的参数
            cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
            g.ndata['cls_feats'][idx] = cls_feats
        else:
            cls_feats = g.ndata['cls_feats'][idx]
        cls_logit = self.classifier(cls_feats)
        cls_pred = th.nn.Softmax(dim=1)(cls_logit)
        gcn_logit = self.gcn(g.ndata['cls_feats'], g, g.edata['edge_weight'])[idx]
        gcn_pred = th.nn.Softmax(dim=1)(gcn_logit)
        pred = (gcn_pred+1e-10) * self.m + cls_pred * (1 - self.m)
        pred = th.log(pred)
        return pred

class BertGCN(th.nn.Module):
    def __init__(self, pretrained_model='roberta_base', nb_class=20, m=0.7, gcn_layers=2, n_hidden=200, dropout=0.5):
        super(BertGCN, self).__init__()
        self.m = m
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = 768
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)
        self.gcn = GCN(
            in_feats=self.feat_dim,
            n_hidden=n_hidden,
            n_classes=nb_class,
            n_layers=gcn_layers-1,
            activation=F.elu,
            dropout=dropout
        )

    def forward(self, g, idx):
        input_ids, attention_mask = g.ndata['input_ids'][idx], g.ndata['attention_mask'][idx]
        if self.training:           # model.train()传过来的参数
            cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
            g.ndata['cls_feats'][idx] = cls_feats
        else:
            cls_feats = g.ndata['cls_feats'][idx]
        cls_logit = self.classifier(cls_feats)
        cls_pred = th.nn.Softmax(dim=1)(cls_logit)
        gcn_logit = self.gcn(g.ndata['cls_feats'], g, g.edata['edge_weight'])[idx]
        gcn_pred = th.nn.Softmax(dim=1)(gcn_logit)
        pred = (gcn_pred+1e-10) * self.m + cls_pred * (1 - self.m)
        pred = th.log(pred)
        return pred
class BertGCNAvg(th.nn.Module):
    def __init__(self, pretrained_model='roberta_base', nb_class=20, m=0.7, gcn_layers=2, n_hidden=200, dropout=0.5):
        super(BertGCNAvg, self).__init__()
        self.m = m
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)
        self.gcn = GCN(
            in_feats=self.feat_dim,
            n_hidden=n_hidden,
            n_classes=nb_class,
            n_layers=gcn_layers-1,
            activation=F.elu,
            dropout=dropout
        )

    def forward(self, g, idx):
        input_ids, attention_mask = g.ndata['input_ids'][idx], g.ndata['attention_mask'][idx]
        if self.training:           # model.train()传过来的参数
            last_hidden_states = self.bert_model(input_ids, attention_mask).last_hidden_state
            cls_feats = last_hidden_states.mean(dim=1)
            g.ndata['cls_feats'][idx] = cls_feats
        else:
            cls_feats = g.ndata['cls_feats'][idx]
        cls_logit = self.classifier(cls_feats)
        cls_pred = th.nn.Softmax(dim=1)(cls_logit)
        gcn_logit = self.gcn(g.ndata['cls_feats'], g, g.edata['edge_weight'])[idx]
        gcn_pred = th.nn.Softmax(dim=1)(gcn_logit)
        pred = (gcn_pred+1e-10) * self.m + cls_pred * (1 - self.m)
        pred = th.log(pred)
        return pred
class BertGCNLda(th.nn.Module):
    def __init__(self, pretrained_model='roberta_base', floats=[], nb_class=20, m=0.7, num_topics=128, gcn_layers=2, n_hidden=200, dropout=0.5):
        super(BertGCNLda, self).__init__()
        self.m = m
        self.num_topics = num_topics
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)
        self.lda_linear = th.nn.Linear(self.num_topics, nb_class)
        self.gcn = GCN(
            in_feats=self.feat_dim,
            n_hidden=n_hidden,
            n_classes=nb_class,
            n_layers=gcn_layers-1,
            activation=F.elu,
            dropout=dropout
        )
        self.floats = floats
    def forward(self, g, idx):
        input_ids, attention_mask, lda_matrix_tensor = g.ndata['input_ids'][idx], g.ndata['attention_mask'][idx], g.ndata['lda_matrix_tensor'][idx]
        if self.training:
            cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
            g.ndata['cls_feats'][idx] = cls_feats
        else:
            cls_feats = g.ndata['cls_feats'][idx]
        cls_logit = self.classifier(cls_feats)
        cls_pred = th.nn.Softmax(dim=1)(cls_logit)
        gcn_logit = self.gcn(g.ndata['cls_feats'], g, g.edata['edge_weight'])[idx]
        gcn_pred = th.nn.Softmax(dim=1)(gcn_logit)
        lda_logit = self.lda_linear(lda_matrix_tensor)
        lda_pred = th.nn.Softmax(dim=1)(lda_logit)

        pred = (gcn_pred+1e-10) * self.floats[0] + cls_pred * self.floats[1] + lda_pred * self.floats[2]
        pred = th.log(pred)
        return pred
class BertGCNLdaAvg(th.nn.Module):
    def __init__(self, pretrained_model='roberta_base', floats=[], nb_class=20, m=0.7, num_topics=128, gcn_layers=2, n_hidden=200, dropout=0.5):
        super(BertGCNLdaAvg, self).__init__()
        self.m = m
        self.num_topics = num_topics
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)
        self.lda_linear = th.nn.Linear(self.num_topics, nb_class)
        self.gcn = GCN(
            in_feats=self.feat_dim,
            n_hidden=n_hidden,
            n_classes=nb_class,
            n_layers=gcn_layers-1,
            activation=F.elu,
            dropout=dropout
        )
        self.floats = floats
    def forward(self, g, idx):
        input_ids, attention_mask, lda_matrix_tensor = g.ndata['input_ids'][idx], g.ndata['attention_mask'][idx], g.ndata['lda_matrix_tensor'][idx]
        if self.training:
            last_hidden_states = self.bert_model(input_ids, attention_mask).last_hidden_state
            cls_feats = last_hidden_states.mean(dim=1)
            g.ndata['cls_feats'][idx] = cls_feats
        else:
            cls_feats = g.ndata['cls_feats'][idx]
        cls_logit = self.classifier(cls_feats)
        cls_pred = th.nn.Softmax(dim=1)(cls_logit)
        gcn_logit = self.gcn(g.ndata['cls_feats'], g, g.edata['edge_weight'])[idx]
        gcn_pred = th.nn.Softmax(dim=1)(gcn_logit)
        lda_logit = self.lda_linear(lda_matrix_tensor)
        lda_pred = th.nn.Softmax(dim=1)(lda_logit)

        pred = (gcn_pred+1e-10) * self.floats[0] + cls_pred * self.floats[1] + lda_pred * self.floats[2]
        pred = th.log(pred)
        return pred
class BertGAT(th.nn.Module):
    def __init__(self, pretrained_model='roberta_base', nb_class=20, m=0.7, gcn_layers=2, heads=8, n_hidden=32, dropout=0.5):
        super(BertGAT, self).__init__()
        self.m = m
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)
        self.gcn = GAT(
                 num_layers=gcn_layers-1,
                 in_dim=self.feat_dim,
                 num_hidden=n_hidden,
                 num_classes=nb_class,
                 heads=[heads] * (gcn_layers-1) + [1],
                 activation=F.elu,
                 feat_drop=dropout,
                 attn_drop=dropout,
        )

    def forward(self, g, idx):
        input_ids, attention_mask = g.ndata['input_ids'][idx], g.ndata['attention_mask'][idx]
        if self.training:
            cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
            g.ndata['cls_feats'][idx] = cls_feats
        else:
            cls_feats = g.ndata['cls_feats'][idx]
        cls_logit = self.classifier(cls_feats)
        cls_pred = th.nn.Softmax(dim=1)(cls_logit)
        gcn_logit = self.gcn(g.ndata['cls_feats'], g)[idx]
        gcn_pred = th.nn.Softmax(dim=1)(gcn_logit)
        pred = (gcn_pred+1e-10) * self.m + cls_pred * (1 - self.m)
        pred = th.log(pred)
        return pred
