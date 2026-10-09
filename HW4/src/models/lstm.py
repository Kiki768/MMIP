import torch.nn as nn

from .base import SequenceClassifier


class LSTMClassifier(SequenceClassifier):
    """LSTM 文字分類器；除了 RNN 換成 LSTM，其餘設定與 RNNClassifier 完全相同。"""

    def __init__(self, vocab_size, **kwargs):
        super().__init__(nn.LSTM, vocab_size, **kwargs)
