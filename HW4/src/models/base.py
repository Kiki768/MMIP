"""RNN / LSTM 文字分類器共用的骨架。

流程：token ids -> Embedding -> (RNN | LSTM) -> 取最後一個「有效」時間步的 hidden state -> Linear

資料是 post-padding（pad 補在句尾），若直接取最後一個時間步會拿到一堆 <pad> 之後的狀態，
所以這裡用 pack_padded_sequence 讓 RNN 只處理真實長度，這樣 RNN 與 LSTM 的比較才公平。
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class SequenceClassifier(nn.Module):
    def __init__(
        self,
        rnn_cls,
        vocab_size,
        embed_dim=128,
        hidden_dim=128,
        num_layers=1,
        num_classes=2,
        dropout=0.3,
        bidirectional=False,
        pad_id=0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.rnn = rnn_cls(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)

    def forward(self, x):
        # <pad> 的 id 固定為 0，真實 token 不會是 0，所以可直接算出每句的有效長度
        lengths = (x != self.pad_id).sum(dim=1).clamp(min=1).cpu()
        emb = self.dropout(self.embedding(x))
        packed = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.rnn(packed)
        if isinstance(hidden, tuple):  # LSTM 回傳 (h_n, c_n)，分類只用 h_n
            hidden = hidden[0]
        if self.bidirectional:
            feat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            feat = hidden[-1]
        return self.fc(self.dropout(feat))
